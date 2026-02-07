# constitution_engine/scripts/run_intake_demo.py
from __future__ import annotations

from datetime import datetime, timezone

from constitution_engine.intake.adapter import draft_episode
from constitution_engine.intake.types import AdapterPolicy, GoalSpec, RawInputItem
from constitution_engine.intake.materialize import materialize_draft_episode
from constitution_engine.invariants.validate import validate_episode
from constitution_engine.runtime.store import ArtifactStore

from constitution_engine.intake.act import act_on_option
from constitution_engine.intake.outcome_log import log_outcome

# Prefer the same StubDrafter import your tests use.
from constitution_engine.intake.stub_drafter import StubDrafter  # type: ignore


def _parse_utc(ts: str) -> datetime:
    """
    Accepts:
      - "2026-02-04T00:00:00Z"
      - "2026-02-04T00:00:00+00:00"
      - naive ISO (treated as UTC)
    """
    s = (ts or "").strip()
    if not s:
        return datetime.now(timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _print_report(report) -> None:
    # report is ValidationReport(subject, violations, resolve_errors)
    print("\n=== VALIDATION REPORT ===")
    print("subject:", report.subject)
    print("ok:", report.ok)
    if report.resolve_errors:
        print("\nresolve_errors:")
        for e in report.resolve_errors:
            print(f"- {e.artifact_type} missing: {e.artifact_id}")
    if report.violations:
        print("\nviolations:")
        for v in report.violations:
            print(f"- {v.rule}: {v.message}")
    if (not report.resolve_errors) and (not report.violations):
        print("(no issues)")


def main() -> None:
    goal = GoalSpec(
        goal_id="g_demo_month",
        statement="Decide whether to build a demo this month.",
        horizon_days=30,
    )

    raw_inputs = [
        RawInputItem(
            raw_id="r1",
            text="I think the engine is solid now, but I'm not sure people will understand it.",
            source_uri="internal:user",
            created_at_utc=_parse_utc("2026-02-04T00:00:00Z"),
        ),
        RawInputItem(
            raw_id="r2",
            text="If I build a demo, it might help me explain it fast. But it could also distract me from finishing the learning loop.",
            source_uri="internal:user",
            created_at_utc=_parse_utc("2026-02-04T00:05:00Z"),
        ),
        RawInputItem(
            raw_id="r3",
            text="I don’t know who the audience is yet (friends? recruiters? builders?).",
            source_uri="internal:user",
            created_at_utc=_parse_utc("2026-02-04T00:08:00Z"),
        ),
    ]

    policy = AdapterPolicy(
        default_confidence=0.6,
        default_uncertainty=0.4,
        auto_probe_on_missing=True,
        allow_commit_proposals=True,
    )

    # -------------------------
    # DRAFT
    # -------------------------
    draft = draft_episode(goal=goal, raw_inputs=raw_inputs, drafter=StubDrafter(), policy=policy)

    print("\n=== DRAFT EPISODE ===")
    print("episode_id:", draft.episode_id)
    print("goal:", draft.goal.statement, f"(horizon_days={draft.goal.horizon_days})")

    print("\n--- Evidence ---")
    for ev in draft.evidence:
        uri = ev.sources[0].uri if getattr(ev, "sources", None) else "(no sources)"
        print(f"- {ev.evidence_id}: {uri} | summary={repr(getattr(ev, 'summary', None))}")

    print("\n--- Observations ---")
    for o in draft.observations:
        max_unc = max((u.level for u in o.uncertainties), default=None)
        print(f"- {o.observation_id} [{o.info_type}] conf={o.confidence.value:.2f} max_unc={max_unc}")
        print(f"  {o.statement}")

    print("\n--- Interpretations ---")
    for it in draft.interpretations:
        max_unc = max((u.level for u in it.uncertainties), default=None)
        print(f"- {it.interpretation_id} [{it.info_type}] conf={it.confidence.value:.2f} max_unc={max_unc}")
        print(f"  title: {it.title}")
        print(f"  narrative: {it.narrative}")

    print("\n--- Options ---")
    for op in draft.options:
        max_unc = max((u.level for u in op.uncertainties), default=None)
        print(
            f"- {op.option_id} kind={op.kind.value} action_class={op.action_class} "
            f"impact={op.impact.value:.2f} rev={op.reversibility.value:.2f} max_unc={max_unc}"
        )
        print(f"  {op.title}: {op.description}")

    print("\n--- Missing Inputs ---")
    if not draft.missing_inputs:
        print("(none)")
    else:
        for mi in draft.missing_inputs:
            print(f"- [{mi.severity}] {mi.field}: {mi.question}")

    print("\n--- Recommendation ---")
    if draft.recommendation is None:
        print("(none)")
    else:
        rec = draft.recommendation
        print("recommendation_id:", rec.recommendation_id)
        print("override_used:", rec.override_used)
        print("override_scope_used:", rec.override_scope_used)
        print("ranked options:")
        for ro in sorted(rec.ranked_options, key=lambda x: x.rank):
            print(f"  #{ro.rank} option_id={ro.option_id} score={ro.score:.2f}")
            print(f"     rationale: {ro.rationale}")

    # -------------------------
    # MATERIALIZE → VALIDATE
    # -------------------------
    store = ArtifactStore()
    episode_id = materialize_draft_episode(store=store, draft=draft)

    print("\n=== MATERIALIZED ===")
    print("stored episode_id:", episode_id)

    report1 = validate_episode(store=store, episode_id=episode_id)
    _print_report(report1)

    # -------------------------
    # PICK OPTION → ACT
    # -------------------------
    chosen_option_id: str | None = None
    chosen_rec_id: str | None = None

    if draft.recommendation and draft.recommendation.ranked_options:
        chosen_option_id = draft.recommendation.top_option_id()
        chosen_rec_id = draft.recommendation.recommendation_id
    else:
        chosen_option_id = draft.options[0].option_id if draft.options else None

    if not chosen_option_id:
        raise RuntimeError("No options available to act on.")

    act_on_option(store=store, episode_id=episode_id, chosen_option_id=chosen_option_id)

    print("\n=== ACTED ===")
    print("chosen_option_id:", chosen_option_id)

    # Validate now (should fail INV-OUT-001 if invariant is active)
    report2 = validate_episode(store=store, episode_id=episode_id)
    _print_report(report2)

    # -------------------------
    # OUTCOME → VALIDATE
    # -------------------------
    out_id = log_outcome(
        store=store,
        episode_id=episode_id,
        recommendation_id=chosen_rec_id,
        chosen_option_id=chosen_option_id,
        description="Ran the chosen plan and captured what happened (demo stub outcome).",
    )

    print("\n=== OUTCOME LOGGED ===")
    print("outcome_id:", out_id)

    report3 = validate_episode(store=store, episode_id=episode_id)
    _print_report(report3)

    # -------------------------
    # REVIEW (only if override used) → VALIDATE
    # -------------------------
    if draft.recommendation and draft.recommendation.override_used:
        from constitution_engine.models.review import ReviewRecord
        from constitution_engine.models.episode import DecisionEpisode
        from constitution_engine.models.types import new_id, now_utc

        review = ReviewRecord(
            review_id=new_id("rev"),
            created_at=now_utc(),
            episode_id=episode_id,
            summary="Override used; reviewed the justification and logged audit entry.",
            override_audit={
                "overrides": [
                    {
                        "recommendation_id": draft.recommendation.recommendation_id,
                        "override_scope_used": list(draft.recommendation.override_scope_used),
                        "rationale": "Thin-slice demo review rationale.",
                    }
                ]
            },
        )
        store.put(review)

        ep_obj = store.must_get(DecisionEpisode, episode_id)
        ep_obj2 = ep_obj.add_reviews(review.review_id)
        store.put(ep_obj2)

        print("\n=== REVIEW LOGGED ===")
        print("review_id:", review.review_id)

        report4 = validate_episode(store=store, episode_id=episode_id)
        _print_report(report4)


if __name__ == "__main__":
    main()
