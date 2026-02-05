from __future__ import annotations

from constitution_engine.intake.drafter import (
    DraftBundle,
    Drafter,
    InterpretationDraft,
    ObservationDraft,
    OptionDraft,
    RecommendationDraft,
)
from constitution_engine.intake.types import AdapterPolicy, GoalSpec, MissingInput, RawInputItem


class StubDrafter(Drafter):
    def draft(
        self,
        *,
        goal: GoalSpec,
        raw_inputs: list[RawInputItem],
        policy: AdapterPolicy,
    ) -> DraftBundle:
        # Minimal, deterministic, safe default bundle.
        return DraftBundle(
            observations=(
                ObservationDraft(
                    statement="The kernel is test-backed and runnable.",
                    confidence=0.7,
                    uncertainty=0.3,
                    info_type="fact",
                ),
            ),
            interpretations=(
                InterpretationDraft(
                    statement="A demo could improve explainability and feedback quality.",
                    confidence=0.6,
                    uncertainty=0.4,
                ),
            ),
            options=(
                OptionDraft(
                    name="Exploratory probe",
                    description="Write a short explainer and ask for feedback; no full demo yet.",
                    impact=0.2,
                    reversibility=0.95,
                    uncertainties=(0.5,),
                    option_kind="info_gathering",
                    action_class="PROBE",
                ),
                OptionDraft(
                    name="Build small demo",
                    description="Create a minimal quick_sim walkthrough; keep scope small.",
                    impact=0.4,
                    reversibility=0.8,
                    uncertainties=(0.4,),
                    option_kind="execute",
                    action_class="LIMITED",
                ),
            ),
            recommendation=RecommendationDraft(
                ranked_option_names=("Exploratory probe", "Build small demo"),
                justification="Uncertainty about audience and opportunity cost remains. Probe first.",
                override_used=False,
            ),
            missing_inputs=(
                MissingInput(
                    field="audience_confirmed",
                    question="Who is the audience and how will we reach them?",
                    severity="MED",
                ),
            ),
        )
