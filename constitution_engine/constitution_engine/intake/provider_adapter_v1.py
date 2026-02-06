# constitution_engine/intake/provider_adapter_v1.py
"""
PROVIDER INTAKE ADAPTER — v1 (FROZEN)

Status: FROZEN
Date: 2026-02-05

Purpose:
- Canonical adapter from (GoalSpec, Evidence[], ProposalSet[]) → DraftEpisode
- Converts provider ProposalSets (LLMs / symbolic reasoners) into kernel-ready draft artifacts
- Enforces provider-boundary invariants before any canonicalization
- Safe-by-default, deterministic, constitution-preserving

Rules:
- DO NOT add intelligence here
- DO NOT add scoring, learning, or heuristics
- DO NOT weaken model invariants to satisfy provider outputs
- Provider outputs are untrusted; reject on boundary violations (strict mode)

If this file needs to change, create:
    intake/provider_adapter_v2.py
and leave this version intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from constitution_engine.invariants.provider_rules import validate_proposalset
from constitution_engine.invariants.rules import InvariantViolation
from constitution_engine.intake.types import AdapterPolicy, DraftEpisode, GoalSpec, MissingInput, RawInputItem

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

# --------------------------------------------------------------------------------------
# Helpers (dict OR dataclass support)
# --------------------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _excerpt(text: str, limit: int = 240) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= limit else (t[: limit - 1] + "…")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _safe_uri(uri: str | None, *, raw_id: str) -> str:
    u = (uri or "").strip()
    return u if u else f"raw://{raw_id}"


def _make_evidence_from_raw(raw_inputs: Iterable[RawInputItem]) -> tuple[Evidence, ...]:
    """
    Convenience: produce Evidence from RawInputItem (same as intake/adapter_v1.py pattern).

    IMPORTANT: If you want providers to reference Evidence IDs, you must:
      - create Evidence first
      - share those evidence_ids with providers
      - then validate provider ProposalSets against them

    This function enables the "create evidence locally" path, but note the above constraint.
    """
    out: list[Evidence] = []
    for ri in raw_inputs:
        excerpt = _excerpt(ri.text, 200)

        src = SourceRef(
            uri=_safe_uri(ri.source_uri, raw_id=ri.raw_id),
            extra={
                "raw_input_id": ri.raw_id,
                "raw_created_at_utc": ri.created_at_utc,
                "adapter": "constitution_engine.intake.provider_adapter_v1",
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


def _normalize_option_kind(kind: str | None) -> OptionKind:
    if not kind:
        return OptionKind.EXECUTE
    v = kind.strip().lower()
    if v in {"info", "info_gathering", "gather", "research"}:
        return OptionKind.INFO_GATHERING
    if v in {"hedge"}:
        return OptionKind.HEDGE
    return OptionKind.EXECUTE


def _normalize_action_class(ac: str | None, policy: AdapterPolicy) -> str:
    """
    Option.action_class is a v0.5.1 bridge field:
    expected values are "probe" | "limited" | "commit".
    """
    if not ac:
        return "probe"
    v = ac.strip().lower()
    if v not in {"probe", "limited", "commit"}:
        return "probe"
    if v == "commit" and not policy.allow_commit_proposals:
        return "limited"
    return v


def _as_uncertainty(level: float, *, description: str) -> Uncertainty:
    return Uncertainty(
        description=description,
        level=_clamp01(level),
        kind=UncertaintyKind.OTHER,
    )


# --------------------------------------------------------------------------------------
# Strict provider validation + canonicalization
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderGateResult:
    accepted: tuple[Any, ...]
    rejected: tuple[tuple[Any, tuple[InvariantViolation, ...]], ...]


def _validate_provider_sets_strict(
    proposal_sets: Sequence[Any],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> ProviderGateResult:
    """
    Strict mode:
      - if a ProposalSet has ANY violations, reject the whole set
      - accepted sets are returned unchanged
    """
    accepted: list[Any] = []
    rejected: list[tuple[Any, tuple[InvariantViolation, ...]]] = []

    # deterministic processing order even if caller passes random order
    ordered = sorted(
        list(proposal_sets),
        key=lambda ps: (
            str(_get(ps, "provider_id", "")),
            str(_get(ps, "model_id", "")),
            str(_get(ps, "run_id", "")),
        ),
    )

    for ps in ordered:
        v = tuple(validate_proposalset(ps, evidence_by_id=evidence_by_id))
        if v:
            rejected.append((ps, v))
        else:
            accepted.append(ps)

    return ProviderGateResult(accepted=tuple(accepted), rejected=tuple(rejected))


def _canonicalize_interpretations(
    proposal_sets: Sequence[Any],
    policy: AdapterPolicy,
    *,
    observation_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> tuple[Interpretation, ...]:
    out: list[Interpretation] = []

    for ps in proposal_sets:
        for it in _as_list(_get(ps, "interpretations", [])):
            text = str(_get(it, "text", "") or "").strip()
            if not text:
                continue

            info_type = InfoType.HYPOTHESIS
            raw_it = _get(it, "info_type", None)
            if raw_it:
                try:
                    info_type = InfoType(str(raw_it))
                except Exception:
                    info_type = InfoType.HYPOTHESIS

            conf_f = float(_get(it, "confidence", policy.default_confidence))

            unc = _get(it, "uncertainty", None)
            unc_level = float(_get(unc, "level", policy.default_uncertainty))

            out.append(
                Interpretation(
                    interpretation_id=new_id("int"),
                    info_type=info_type,
                    title=_excerpt(text, 80),
                    narrative=text,
                    confidence=Confidence(_clamp01(conf_f)),
                    uncertainties=(_as_uncertainty(unc_level, description="provider interpretation uncertainty"),),
                    observation_ids=observation_ids,
                    evidence_ids=evidence_ids,
                    created_at=now_utc(),
                )
            )

    return tuple(out)


def _canonicalize_options(
    proposal_sets: Sequence[Any],
    policy: AdapterPolicy,
    *,
    observation_ids: tuple[str, ...],
    interpretation_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> tuple[Option, ...]:
    """
    Canonicalize provider-proposed options into kernel Option artifacts.

    Determinism:
      - options are deduped by a stable key (kind, action_class, title, description)
      - then sorted by that key to yield stable IDs & ordering
    """
    raw_opts: list[dict[str, Any]] = []

    for ps in proposal_sets:
        for opt in _as_list(_get(ps, "options", [])):
            title = str(_get(opt, "title", "") or "").strip()
            desc = str(_get(opt, "description", "") or "").strip()
            if not title:
                continue
            raw_opts.append(
                {
                    "provider_option_id": str(_get(opt, "option_id", "") or ""),
                    "kind": str(_get(opt, "kind", "") or ""),
                    "action_class": str(_get(opt, "action_class", "") or ""),
                    "title": title,
                    "description": desc,
                    "impact": float(_get(opt, "impact", 0.3)),
                    "reversibility": float(_get(opt, "reversibility", 0.8)),
                    "unc_level": float(_get(_get(opt, "uncertainty", None), "level", policy.default_uncertainty)),
                }
            )

    def k(d: dict[str, Any]) -> tuple[str, str, str, str]:
        kind = _normalize_option_kind(d["kind"]).value
        ac = _normalize_action_class(d["action_class"], policy)
        return (kind, ac, d["title"].lower(), d["description"].lower())

    # dedupe deterministically
    dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for d in raw_opts:
        dedup.setdefault(k(d), d)

    ordered = sorted(dedup.values(), key=k)

    out: list[Option] = []
    for d in ordered:
        kind = _normalize_option_kind(d["kind"])
        ac = _normalize_action_class(d["action_class"], policy)

        out.append(
            Option(
                option_id=new_id("opt"),
                kind=kind,
                title=d["title"],
                description=d["description"],
                action_class=ac,
                impact=Impact(_clamp01(float(d["impact"]))),
                reversibility=Reversibility(_clamp01(float(d["reversibility"]))),
                uncertainties=(_as_uncertainty(float(d["unc_level"]), description="provider option uncertainty"),),
                observation_ids=observation_ids,
                interpretation_ids=interpretation_ids,
                evidence_ids=evidence_ids,
                created_at=now_utc(),
            )
        )

    return tuple(out)


def _select_primary_ranking_source(proposal_sets: Sequence[Any]) -> Any | None:
    """
    Deterministic selection of a primary ranking source:
      - first ProposalSet (already ordered upstream) that includes ranked_options
    """
    for ps in proposal_sets:
        ros = _as_list(_get(ps, "ranked_options", []))
        if ros:
            return ps
    return None


def _build_provider_option_title_index(proposal_set: Any) -> Mapping[str, str]:
    """
    Map provider option_id -> provider option.title for the given ProposalSet.
    Used to resolve ranked_options.option_ref when ranked option has no explicit title.
    """
    idx: dict[str, str] = {}
    for opt in _as_list(_get(proposal_set, "options", [])):
        pid = str(_get(opt, "option_id", "") or "").strip()
        title = str(_get(opt, "title", "") or "").strip()
        if pid and title and pid not in idx:
            idx[pid] = title
    return idx


def _canonicalize_recommendation(
    proposal_sets: Sequence[Any],
    policy: AdapterPolicy,
    *,
    options: tuple[Option, ...],
    orientation_id: str,
    evidence_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    interpretation_ids: tuple[str, ...],
) -> Recommendation | None:
    """
    Construct a canonical Recommendation using provider rankings if present.

    Notes:
    - The kernel owns Recommendation semantics; providers supply ranking proposals.
    - Scoring is NOT performed here; scores are placeholders.
    - Mapping strategy (deterministic, v1):
        1) Prefer ranked_option.title (if present)
        2) Else use ranked_option.option_ref -> ProposalSet.options[option_id].title
        3) Map title -> canonical option_id (post-dedup)
    """
    if not options:
        return None

    # Map by title to canonical option_id (deterministic since options are deterministic)
    by_title = {o.title: o.option_id for o in options}

    ps = _select_primary_ranking_source(proposal_sets)
    if ps is None:
        # No provider ranking → do not emit recommendation at draft stage (safe-by-default).
        return None

    provider_title_by_option_id = _build_provider_option_title_index(ps)
    ros = _as_list(_get(ps, "ranked_options", []))

    ranked: list[RankedOption] = []
    rank = 1

    # Sort by provider rank value; keep stable fallback for missing/invalid.
    def _rank_key(r: Any) -> int:
        try:
            return int(_get(r, "rank", 10**9))
        except Exception:
            return 10**9

    for ro in sorted(ros, key=_rank_key):
        # Resolve title:
        title = str(_get(ro, "title", "") or "").strip()
        if not title:
            opt_ref = str(_get(ro, "option_ref", "") or "").strip()
            title = provider_title_by_option_id.get(opt_ref, "")

        if not title:
            # Cannot map deterministically; skip.
            continue

        oid = by_title.get(title)
        if not oid:
            continue

        rationale = str(_get(ro, "rationale", "") or "").strip()
        if not rationale:
            rationale = "Provider ranking proposal."

        conf_f = float(_get(ro, "confidence", policy.default_confidence))

        unc = _get(ro, "uncertainty", None)
        unc_level = float(_get(unc, "level", policy.default_uncertainty))

        ranked.append(
            RankedOption(
                option_id=oid,
                rank=rank,
                score=0.5,  # adapter does not score
                rationale=rationale,
                confidence=Confidence(_clamp01(conf_f)),
                uncertainties=(_as_uncertainty(unc_level, description="provider ranked_option uncertainty"),),
            )
        )
        rank += 1

    if not ranked:
        return None

    return Recommendation(
        recommendation_id=new_id("rec"),
        created_at=now_utc(),
        orientation_id=orientation_id,
        ranked_options=tuple(ranked),
        evidence_ids=evidence_ids,
        observation_ids=observation_ids,
        interpretation_ids=interpretation_ids,
        override_used=False,
        override_scope_used=(),
        uncertainty_summary="Provider ranking proposal (adapter does not score).",
        summary="Provider ranking proposal.",
        meta={"adapter": "constitution_engine.intake.provider_adapter_v1"},
    )


# --------------------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------------------


def draft_episode_from_proposals(
    *,
    goal: GoalSpec,
    raw_inputs: list[RawInputItem] | None,
    proposal_sets: Sequence[Any],
    policy: AdapterPolicy = AdapterPolicy(),
    evidence: tuple[Evidence, ...] | None = None,
) -> DraftEpisode:
    """
    Provider path: (Goal, Evidence?, ProposalSet[]) -> DraftEpisode

    Usage pattern (recommended):
      1) Create Evidence (either here from raw_inputs, or externally and pass `evidence=...`)
      2) Run providers *with evidence_ids visible to them*
      3) Validate ProposalSets against evidence_by_id
      4) Canonicalize provider proposals into DraftEpisode artifacts

    Strict mode:
      - any ProposalSet with violations is rejected in full
      - rejected sets do NOT block episode creation (safe fallback behavior)
    """
    if evidence is None:
        if raw_inputs is None:
            raw_inputs = []
        evidence = _make_evidence_from_raw(raw_inputs)

    evidence_by_id = {ev.evidence_id: ev for ev in evidence}
    ev_ids = tuple(ev.evidence_id for ev in evidence)

    gate = _validate_provider_sets_strict(proposal_sets, evidence_by_id=evidence_by_id)

    # For v1, we do not create Observations from providers (providers propose interpretations/options).
    # Keep observations empty; provenance is still threaded via evidence_ids.
    observations: tuple[Observation, ...] = ()
    obs_ids: tuple[str, ...] = ()

    interpretations = _canonicalize_interpretations(
        gate.accepted,
        policy,
        observation_ids=obs_ids,
        evidence_ids=ev_ids,
    )
    int_ids = tuple(i.interpretation_id for i in interpretations)

    options = _canonicalize_options(
        gate.accepted,
        policy,
        observation_ids=obs_ids,
        interpretation_ids=int_ids,
        evidence_ids=ev_ids,
    )

    # DraftEpisode does not include Orientation artifact; use placeholder non-empty.
    ori_id = new_id("ori_draft")

    recommendation = _canonicalize_recommendation(
        gate.accepted,
        policy,
        options=options,
        orientation_id=ori_id,
        evidence_ids=ev_ids,
        observation_ids=obs_ids,
        interpretation_ids=int_ids,
    )

    # Missing inputs are not inferred here (no intelligence). Keep empty.
    missing_inputs: tuple[MissingInput, ...] = ()

    # Notes: keep minimal; record rejection count deterministically (no sensitive details).
    notes: tuple[str, ...] = ()
    if gate.rejected:
        notes = (f"provider_sets_rejected={len(gate.rejected)}",)

    return DraftEpisode(
        episode_id=new_id("ep"),
        goal=goal,
        evidence=evidence,
        observations=observations,
        interpretations=interpretations,
        options=options,
        recommendation=recommendation,
        missing_inputs=missing_inputs,
        notes=notes,
    )
