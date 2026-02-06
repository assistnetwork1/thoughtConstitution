"""
INTAKE ADAPTER — FROZEN (v1)

Status: FROZEN
Date: 2026-02-04

Purpose:
- Canonical adapter from (GoalSpec, RawInputItem[]) → DraftEpisode
- Converts messy inputs into typed constitutional atoms
- Safe-by-default, uncertainty-first, governance-correct

Rules:
- DO NOT add intelligence here
- DO NOT add scoring, learning, or heuristics
- DO NOT weaken model invariants to satisfy drafts
- Extensions MUST occur in:
    - Drafter implementations
    - Orientation construction
    - Review / Learning layers

If this file needs to change, create:
    intake/adapter_v2.py
and leave this version intact.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from constitution_engine.intake.drafter import (
    DraftBundle,
    Drafter,
    OptionDraft,
    RecommendationDraft,
)
from constitution_engine.intake.types import (
    AdapterPolicy,
    DraftEpisode,
    GoalSpec,
    MissingInput,
    RawInputItem,
)
from constitution_engine.models.evidence import Evidence, SourceRef, SpanRef
from constitution_engine.models.interpretation import Interpretation
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import RankedOption, Recommendation
from constitution_engine.models.types import (
    Confidence,
    Impact,
    InfoType,
    Reversibility,
    Uncertainty,
    UncertaintyKind,
    new_id,
    now_utc,
)


def _excerpt(text: str, limit: int = 240) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= limit else (t[: limit - 1] + "…")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _fill_conf_unc(
    *,
    confidence: float | None,
    uncertainty: float | None,
    policy: AdapterPolicy,
) -> tuple[float, float]:
    c = policy.default_confidence if confidence is None else confidence
    u = policy.default_uncertainty if uncertainty is None else uncertainty
    return (_clamp01(c), _clamp01(u))


def _safe_uri(uri: str | None, *, raw_id: str) -> str:
    u = (uri or "").strip()
    return u if u else f"raw://{raw_id}"


def _make_evidence(raw_inputs: Iterable[RawInputItem]) -> tuple[Evidence, ...]:
    """
    Evidence is provenance. In this repo:
      - SourceRef contains source metadata only (no span/excerpt field).
      - Evidence holds sources + spans + summary/notes, plus integrity confidence.

    We store a short excerpt in Evidence.summary and attach raw_input metadata in SourceRef.extra.
    """
    out: list[Evidence] = []
    for ri in raw_inputs:
        excerpt = _excerpt(ri.text, 200)

        src = SourceRef(
            uri=_safe_uri(ri.source_uri, raw_id=ri.raw_id),
            extra={
                "raw_input_id": ri.raw_id,
                "raw_created_at_utc": ri.created_at_utc,
                "adapter": "constitution_engine.intake.adapter",
            },
        )

        spans: tuple[SpanRef, ...] = ()

        out.append(
            Evidence(
                evidence_id=new_id("ev"),
                created_at=now_utc(),
                sources=(src,),
                spans=spans,
                summary=excerpt,
                notes={"raw_text_len": len(ri.text or "")},
                integrity=Confidence(1.0),
            )
        )
    return tuple(out)


def _make_observations(
    bundle: DraftBundle,
    policy: AdapterPolicy,
    *,
    raw_input_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> tuple[Observation, ...]:
    """
    Observation model (per your repo):
      - confidence: Confidence
      - uncertainties: Sequence[Uncertainty]
      - provenance slots: raw_input_ids, evidence_ids
    """
    out: list[Observation] = []
    for od in bundle.observations:
        conf_f, unc_f = _fill_conf_unc(confidence=od.confidence, uncertainty=od.uncertainty, policy=policy)

        info_type = InfoType.FACT
        if getattr(od, "info_type", None):
            try:
                info_type = InfoType(od.info_type)  # type: ignore[arg-type]
            except Exception:
                info_type = InfoType.FACT

        unc_obj = Uncertainty(
            description="draft intake uncertainty",
            level=unc_f,
            kind=UncertaintyKind.OTHER,
        )

        statement = (od.statement or "").strip()
        if not statement:
            # Adapter is allowed to be conservative: drop empty statements rather than force garbage.
            continue

        out.append(
            Observation(
                observation_id=new_id("obs"),
                statement=statement,
                info_type=info_type,
                confidence=Confidence(conf_f),
                uncertainties=(unc_obj,),
                raw_input_ids=raw_input_ids,
                evidence_ids=evidence_ids,
                created_at=now_utc(),
            )
        )
    return tuple(out)


def _make_interpretations(
    bundle: DraftBundle,
    policy: AdapterPolicy,
    *,
    observation_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> tuple[Interpretation, ...]:
    """
    Interpretation model (per your repo):
      - info_type must be interpretive
      - title, narrative (no 'statement' field)
      - confidence: Confidence
      - uncertainties: Sequence[Uncertainty]
      - provenance slots: observation_ids, evidence_ids
    """
    out: list[Interpretation] = []
    for itd in bundle.interpretations:
        conf_f, unc_f = _fill_conf_unc(confidence=itd.confidence, uncertainty=itd.uncertainty, policy=policy)

        info_type = InfoType.HYPOTHESIS
        if getattr(itd, "info_type", None):
            try:
                info_type = InfoType(itd.info_type)  # type: ignore[arg-type]
            except Exception:
                info_type = InfoType.HYPOTHESIS

        text = (itd.statement or "").strip()
        if not text:
            continue

        title = _excerpt(text, 80)
        narrative = text

        unc_obj = Uncertainty(
            description="draft intake uncertainty",
            level=unc_f,
            kind=UncertaintyKind.OTHER,
        )

        out.append(
            Interpretation(
                interpretation_id=new_id("int"),
                info_type=info_type,
                title=title,
                narrative=narrative,
                confidence=Confidence(conf_f),
                uncertainties=(unc_obj,),
                observation_ids=observation_ids,
                evidence_ids=evidence_ids,
                created_at=now_utc(),
            )
        )
    return tuple(out)


def _normalize_action_class(ac: str | None, policy: AdapterPolicy) -> str:
    """
    Option.action_class is a v0.5.1 bridge field:
    expected values are "probe" | "limited" | "commit".
    We normalize to lowercase.
    """
    if not ac:
        return "probe"
    val = ac.strip().lower()
    if val not in {"probe", "limited", "commit"}:
        return "probe"
    if val == "commit" and not policy.allow_commit_proposals:
        return "limited"
    return val


def _normalize_option_kind(ok: str | None) -> OptionKind:
    if not ok:
        return OptionKind.EXECUTE
    val = ok.strip().lower()
    if val in {"info", "info_gathering", "gather", "research"}:
        return OptionKind.INFO_GATHERING
    if val in {"hedge"}:
        return OptionKind.HEDGE
    return OptionKind.EXECUTE


def _make_options(
    bundle: DraftBundle,
    policy: AdapterPolicy,
    *,
    observation_ids: tuple[str, ...],
    interpretation_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> tuple[Option, ...]:
    """
    Option model (per your repo):
      - title (not name)
      - impact: Impact wrapper
      - reversibility: Reversibility wrapper
      - uncertainties: Sequence[Uncertainty] objects
      - upstream references: observation_ids, interpretation_ids, evidence_ids
    """
    out: list[Option] = []
    for op in bundle.options:
        name = (op.name or "").strip()
        desc = (op.description or "").strip()
        if not name:
            continue

        impact_f = _clamp01(op.impact if op.impact is not None else 0.3)
        rev_f = _clamp01(op.reversibility if op.reversibility is not None else 0.8)

        ac = _normalize_action_class(op.action_class, policy)
        kind = _normalize_option_kind(op.option_kind)

        unc_levels = op.uncertainties or (policy.default_uncertainty,)
        unc_objs = tuple(
            Uncertainty(
                description="draft option uncertainty",
                level=_clamp01(u),
                kind=UncertaintyKind.OTHER,
            )
            for u in unc_levels
        )

        out.append(
            Option(
                option_id=new_id("opt"),
                kind=kind,
                title=name,
                description=desc,
                action_class=ac,
                impact=Impact(impact_f),
                reversibility=Reversibility(rev_f),
                uncertainties=unc_objs,
                observation_ids=observation_ids,
                interpretation_ids=interpretation_ids,
                evidence_ids=evidence_ids,
                created_at=now_utc(),
            )
        )
    return tuple(out)


def _auto_probe_options(missing: tuple[MissingInput, ...], policy: AdapterPolicy) -> tuple[OptionDraft, ...]:
    if not policy.auto_probe_on_missing:
        return ()
    severe = any(m.severity in {"HIGH", "MED"} for m in missing)
    if not severe:
        return ()
    return (
        OptionDraft(
            name="Clarify unknowns",
            description="Gather missing inputs before any irreversible action.",
            impact=0.2,
            reversibility=0.95,
            uncertainties=(0.6,),
            option_kind="info_gathering",
            action_class="PROBE",
        ),
    )


def _make_recommendation(
    rec_draft: RecommendationDraft | None,
    *,
    options: tuple[Option, ...],
    orientation_id: str,
    evidence_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    interpretation_ids: tuple[str, ...],
) -> Recommendation | None:
    """
    Recommendation model (per your repo):
      - orientation_id is REQUIRED and must be non-empty
      - ranked_options require: option_id, rank, score, rationale
      - justification fields are permissive: use summary / uncertainty_summary / meta
    """
    if rec_draft is None:
        return None

    by_title = {o.title: o.option_id for o in options}

    ranked: list[RankedOption] = []
    rank = 1
    for title in rec_draft.ranked_option_names:
        oid = by_title.get(title)
        if not oid:
            continue
        ranked.append(
            RankedOption(
                option_id=oid,
                rank=rank,
                score=0.5,  # neutral placeholder (adapter is not a scorer)
                rationale="Draft ranking from intake adapter (no scoring model applied yet).",
                confidence=Confidence(0.6),
                uncertainties=(
                    Uncertainty(
                        description="draft recommendation ranking uncertainty",
                        level=0.5,
                        kind=UncertaintyKind.OTHER,
                    ),
                ),
            )
        )
        rank += 1

    # If drafter gave us nothing usable, don't emit a Recommendation (keeps things safe).
    if not ranked:
        return None

    justification = (rec_draft.justification or "").strip()
    if not justification:
        justification = "Draft recommendation emitted from intake adapter."

    return Recommendation(
        recommendation_id=new_id("rec"),
        created_at=now_utc(),
        orientation_id=orientation_id,
        ranked_options=tuple(ranked),
        evidence_ids=evidence_ids,
        observation_ids=observation_ids,
        interpretation_ids=interpretation_ids,
        override_used=bool(rec_draft.override_used),
        override_scope_used=tuple(rec_draft.override_scope_used),
        uncertainty_summary=justification,
        summary=justification,
        meta={"adapter": "constitution_engine.intake.adapter"},
    )


def draft_episode(
    *,
    goal: GoalSpec,
    raw_inputs: list[RawInputItem],
    drafter: Drafter,
    policy: AdapterPolicy = AdapterPolicy(),
) -> DraftEpisode:
    """
    (Goal, RawInput) -> DraftEpisode

    Responsibilities:
    - Create Evidence from RawInput (provenance anchor)
    - Ask Drafter for DraftBundle (draft atoms)
    - Normalize drafts into canonical model artifacts with safe defaults
    - Auto-insert PROBE info-gathering option if key inputs are missing
    """
    evidence = _make_evidence(raw_inputs)

    # Provenance IDs for linking
    raw_ids = tuple(ri.raw_id for ri in raw_inputs)
    ev_ids = tuple(ev.evidence_id for ev in evidence)

    bundle = drafter.draft(goal=goal, raw_inputs=raw_inputs, policy=policy)

    extra = _auto_probe_options(tuple(bundle.missing_inputs), policy)
    if extra:
        bundle = replace(bundle, options=tuple(bundle.options) + tuple(extra))

    observations = _make_observations(bundle, policy, raw_input_ids=raw_ids, evidence_ids=ev_ids)
    obs_ids = tuple(o.observation_id for o in observations)

    interpretations = _make_interpretations(bundle, policy, observation_ids=obs_ids, evidence_ids=ev_ids)
    int_ids = tuple(i.interpretation_id for i in interpretations)

    options = _make_options(
        bundle,
        policy,
        observation_ids=obs_ids,
        interpretation_ids=int_ids,
        evidence_ids=ev_ids,
    )

    # DraftEpisode currently doesn't include an Orientation artifact.
    # But Recommendation requires a non-empty orientation_id, so we generate a draft placeholder.
    ori_id = new_id("ori_draft")

    recommendation = _make_recommendation(
        bundle.recommendation,
        options=options,
        orientation_id=ori_id,
        evidence_ids=ev_ids,
        observation_ids=obs_ids,
        interpretation_ids=int_ids,
    )

    return DraftEpisode(
        episode_id=new_id("ep"),
        goal=goal,
        evidence=evidence,
        observations=observations,
        interpretations=interpretations,
        options=options,
        recommendation=recommendation,
        missing_inputs=tuple(bundle.missing_inputs),
        notes=(),
    )
