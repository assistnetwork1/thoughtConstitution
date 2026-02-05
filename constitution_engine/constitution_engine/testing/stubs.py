from __future__ import annotations

from constitution_engine.intake.drafter import (
    DraftBundle,
    Drafter,
    ObservationDraft,
    InterpretationDraft,
    OptionDraft,
    RecommendationDraft,
)
from constitution_engine.intake.types import AdapterPolicy, GoalSpec, RawInputItem


class StubDrafter(Drafter):
    """
    Deterministic stub used by smoke tests.

    Intent:
    - Always returns at least:
        * 1 ObservationDraft
        * 1 InterpretationDraft
        * 2 OptionDrafts
        * 1 MissingInput (MED)
        * 1 RecommendationDraft ranking the two options
    - Does NOT depend on any external services or store.
    """

    def draft(self, *, goal: GoalSpec, raw_inputs: list[RawInputItem], policy: AdapterPolicy) -> DraftBundle:
        # Keep it simple and stable: one clear observation, one clear hypothesis,
        # then two actionable options + a missing-input prompt.
        observations = (
            ObservationDraft(
                statement="The kernel is test-backed and runnable.",
                confidence=0.7,
                uncertainty=0.3,
                info_type="fact",
            ),
        )

        interpretations = (
            InterpretationDraft(
                statement="A demo could improve explainability and feedback quality.",
                confidence=0.6,
                uncertainty=0.4,
            ),
        )

        options = (
            OptionDraft(
                name="Exploratory probe",
                description="Write a short explainer and ask for feedback; no full demo yet.",
                impact=0.2,
                reversibility=0.95,
                uncertainties=(0.5,),
                option_kind="info_gathering",
                action_class="probe",
            ),
            OptionDraft(
                name="Build small demo",
                description="Create a minimal quick_sim walkthrough; keep scope small.",
                impact=0.4,
                reversibility=0.8,
                uncertainties=(0.4,),
                option_kind="execute",
                action_class="limited",
            ),
        )

        recommendation = RecommendationDraft(
            ranked_option_names=("Exploratory probe", "Build small demo"),
            justification="Uncertainty about audience and opportunity cost remains. Probe first.",
            override_used=False,
            override_scope_used=(),
        )

        return DraftBundle(
            observations=observations,
            interpretations=interpretations,
            options=options,
            recommendation=recommendation,
            missing_inputs=(
                # Match your earlier demo output
                # (field names + severity strings are the ones your intake adapter expects)
                # If your MissingInput type is in drafter.py instead, import it from there.
                # Otherwise, keep this empty and let adapter auto-probe.
                # NOTE: DraftBundle.missing_inputs expects MissingInput objects (from intake.types).
                # We import it lazily to avoid circular imports in some layouts.
                __import__("constitution_engine.intake.types", fromlist=["MissingInput"]).MissingInput(
                    field="audience_confirmed",
                    question="Who is the audience and how will we reach them?",
                    severity="MED",
                ),
            ),
        )
