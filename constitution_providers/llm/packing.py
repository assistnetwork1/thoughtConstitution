# constitution_providers/llm/packing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PromptPack:
    """
    v1 prompt pack: string templates only (no Jinja yet).
    Later you can extend to filesystem-backed packs.
    """
    pack_id: str
    pack_version: str
    system_template: str
    user_template: str


@dataclass(frozen=True)
class RenderedPrompt:
    """
    Canonical, deterministic prompt object for adapters.
    """
    pack_id: str
    pack_version: str
    system: str
    user: str
    meta: Mapping[str, object]


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _stable_lines(items: Sequence[str]) -> str:
    """
    Deterministic formatting: keep given order, strip, drop empties.
    """
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        if s:
            out.append(s)
    return "\n".join(out)


def _extract_orientation_fields(orientation: Any) -> tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
    """
    Support both:
      - Orientation(objectives=..., constraints=..., values=...)
      - Orientation().add_objectives(...) patterns where objectives might be objects

    Returns raw sequences (not stringified yet).
    """
    if orientation is None:
        return (), (), ()

    objectives = getattr(orientation, "objectives", ()) or ()
    constraints = getattr(orientation, "constraints", ()) or ()
    values = getattr(orientation, "values", ()) or ()
    return objectives, constraints, values


def _stringify_objectives(objs: Sequence[Any]) -> str:
    """
    Objective may be:
      - str
      - Objective(name=..., description=..., weight=...)
      - any object with .name/.description
    We render a stable bullet-like format.
    """
    lines: list[str] = []
    for o in objs:
        if o is None:
            continue
        if isinstance(o, str):
            lines.append(o)
            continue

        name = getattr(o, "name", None)
        desc = getattr(o, "description", None)

        # If it looks like an Objective dataclass/object, format it.
        if name or desc:
            name_s = _safe_str(name).strip()
            desc_s = _safe_str(desc).strip()
            if name_s and desc_s:
                lines.append(f"{name_s}: {desc_s}")
            elif name_s:
                lines.append(name_s)
            elif desc_s:
                lines.append(desc_s)
            continue

        # fallback
        lines.append(_safe_str(o))

    return _stable_lines([_safe_str(x) for x in lines])


def _stringify_scalars(items: Sequence[Any]) -> str:
    return _stable_lines([_safe_str(x) for x in items])


def _stringify_raw_inputs(raw_inputs: Sequence[Any]) -> str:
    """
    RawInput may have:
      - .payload (your kernel)
      - .text (alternate)
    We render each payload as a separate line.
    """
    vals: list[str] = []
    for ri in raw_inputs:
        if ri is None:
            continue
        payload = getattr(ri, "payload", None)
        if payload is None:
            payload = getattr(ri, "text", "")
        vals.append(_safe_str(payload))
    return _stable_lines(vals)


def _stringify_evidence(evidence: Sequence[Any]) -> str:
    """
    Evidence may have:
      - .summary
      - .sources (with uri/title)
    We primarily include summary text (deterministic).
    """
    ev_summaries: list[str] = []
    for ev in evidence:
        if ev is None:
            continue
        s = _safe_str(getattr(ev, "summary", ""))
        s = s.strip()
        if s:
            ev_summaries.append(s)
    return _stable_lines(ev_summaries)


def _stringify_evidence_ids(evidence: Sequence[Any]) -> str:
    """
    Deterministic list of allowed evidence IDs.
    Providers MUST ONLY reference these IDs in evidence_refs.
    """
    ids: list[str] = []
    for ev in evidence:
        if ev is None:
            continue
        ev_id = getattr(ev, "evidence_id", None)
        if isinstance(ev_id, str) and ev_id.strip():
            ids.append(ev_id.strip())
    return _stable_lines(ids)


def render_prompt(
    *,
    pack: PromptPack,
    ctx: Any,
    extra: Mapping[str, object] | None = None,
) -> RenderedPrompt:
    """
    Deterministic packing of EpisodeContext → prompt text.

    Rules:
    - No intelligence here.
    - Only formatting + threading of context fields.
    - Keep it stable across runs.

    EpisodeContext contract (expected):
      ctx.orientation (has .objectives / .constraints / .values or similar)
      ctx.raw_inputs (sequence; each has .payload or .text)
      ctx.evidence (sequence; each has .summary and/or sources)
    """
    orientation = getattr(ctx, "orientation", None)
    objectives, constraints, values = _extract_orientation_fields(orientation)

    raw_inputs = getattr(ctx, "raw_inputs", ()) or ()
    evidence = getattr(ctx, "evidence", ()) or ()

    obj_txt = _stringify_objectives(objectives)
    con_txt = _stringify_scalars(constraints)
    val_txt = _stringify_scalars(values)

    raw_txt = _stringify_raw_inputs(raw_inputs)
    ev_txt = _stringify_evidence(evidence)
    ev_ids_txt = _stringify_evidence_ids(evidence)

    # Basic templating (v1). Keep templates pure; no arbitrary code.
    system = pack.system_template.format(
        pack_id=pack.pack_id,
        pack_version=pack.pack_version,
    )

    # IMPORTANT: user_template may include literal braces (e.g., JSON examples).
    # Any literal { or } must be escaped in the template as {{ or }}.
    user = pack.user_template.format(
        objectives=obj_txt,
        constraints=con_txt,
        values=val_txt,
        raw_inputs=raw_txt,
        evidence=ev_txt,
        evidence_ids=ev_ids_txt,
    )

    meta: dict[str, object] = {
        "pack_id": pack.pack_id,
        "pack_version": pack.pack_version,
        "objective_count": len(objectives),
        "raw_input_count": len(raw_inputs),
        "evidence_count": len(evidence),
        "evidence_ids": tuple(
            [line.strip() for line in (ev_ids_txt.splitlines() if ev_ids_txt else []) if line.strip()]
        ),
    }
    if extra:
        meta.update(dict(extra))

    return RenderedPrompt(
        pack_id=pack.pack_id,
        pack_version=pack.pack_version,
        system=system,
        user=user,
        meta=meta,
    )


def default_reasoner_pack_v1() -> PromptPack:
    """
    A conservative, JSON-only pack.

    NOTE:
    - This pack uses str.format() templating.
    - Therefore, any literal JSON braces in the template must be escaped as {{ and }}.
    """
    system = (
        "You are a constitutional reasoning provider.\n"
        "You MUST output ONLY valid JSON.\n"
        "Do NOT include markdown.\n"
        "Do NOT output kernel-owned artifacts (Recommendation/Choice/Outcome/Review/etc.).\n"
        "Return proposals only.\n"
        "\n"
        "Hard constraints:\n"
        "- ranked_options must be a strict total order with ranks 1..N\n"
        '- option_ref must reference an option_id from "options"\n'
        "- Every Interpretation/Option/RankedOption/OverrideSuggestion must include:\n"
        "  confidence (number), uncertainty.level (number), evidence_refs (list), limits (string)\n"
        "\n"
        "Evidence constraint:\n"
        "- evidence_refs MUST ONLY contain IDs listed in VALID EVIDENCE IDS.\n"
    )

    user = (
        "CONTEXT\n"
        "Objectives:\n{objectives}\n\n"
        "Constraints:\n{constraints}\n\n"
        "Values:\n{values}\n\n"
        "Raw Inputs:\n{raw_inputs}\n\n"
        "Evidence Summaries:\n{evidence}\n\n"
        "VALID EVIDENCE IDS (you MUST ONLY reference these in evidence_refs):\n"
        "{evidence_ids}\n\n"
        "TASK\n"
        "Propose:\n"
        "1) interpretations (hypothesis/explanation/frame) linked to evidence_refs\n"
        "2) options (probe/limited/commit proposal only) linked to evidence_refs\n"
        "3) ranked_options referencing option_ref\n\n"
        "OUTPUT SCHEMA (JSON ONLY)\n"
        "{{\n"
        '  "interpretations": [\n'
        "    {{\n"
        '      "interpretation_id": "prov_int_1",\n'
        '      "info_type": "hypothesis",\n'
        '      "text": "...",\n'
        '      "confidence": 0.0,\n'
        '      "uncertainty": {{"level": 0.0}},\n'
        '      "evidence_refs": ["<choose from VALID EVIDENCE IDS>"],\n'
        '      "limits": "..." \n'
        "    }}\n"
        "  ],\n"
        '  "options": [\n'
        "    {{\n"
        '      "option_id": "prov_opt_1",\n'
        '      "kind": "info_gathering",\n'
        '      "title": "...",\n'
        '      "description": "...",\n'
        '      "action_class": "probe",\n'
        '      "impact": 0.0,\n'
        '      "reversibility": 0.0,\n'
        '      "confidence": 0.0,\n'
        '      "uncertainty": {{"level": 0.0}},\n'
        '      "evidence_refs": ["<choose from VALID EVIDENCE IDS>"],\n'
        '      "limits": "..."\n'
        "    }}\n"
        "  ],\n"
        '  "ranked_options": [\n'
        "    {{\n"
        '      "rank": 1,\n'
        '      "option_ref": "prov_opt_1",\n'
        '      "rationale": "...",\n'
        '      "title": "...",\n'
        '      "confidence": 0.5,\n'
        '      "uncertainty": {{"level": 0.5}},\n'
        '      "evidence_refs": ["<choose from VALID EVIDENCE IDS>"],\n'
        '      "limits": "..." \n'
        "    }}\n"
        "  ],\n"
        '  "override_suggestions": []\n'
        "}}\n"
    )

    return PromptPack(
        pack_id="reasoner",
        pack_version="v1",
        system_template=system,
        user_template=user,
    )
