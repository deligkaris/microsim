from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.cv_model_repository import CVPrevalenceModelRepository
from microsim.outcomes.stroke_partition_model_repository import StrokePrevalenceModelRepository
from microsim.outcomes.mi_partition_model_repository import MIPrevalenceModelRepository
from microsim.outcomes.dementia_model_repository import DementiaPrevalenceModelRepository
from microsim.outcomes.diabetes_model_repository import DiabetesPrevalenceModelRepository
from microsim.outcomes.chronic_kidney_disease_model_repository import (
    ChronicKidneyDiseasePrevalenceModelRepository,
)
from microsim.outcomes.epilepsy_model_repository import EpilepsyPrevalenceModelRepository


class OutcomePrevalenceModelRepository:
    """Holds the priorToSim (prevalence) rules for outcomes.
       Mirror image of OutcomeModelRepository: every OutcomeType is registered as a key, but
       outcomes without a prevalence model resolve to None and are skipped during seeding."""

    def __init__(self):
        self._repository = {
            OutcomeType.WMH:                      None,
            OutcomeType.COGNITION:                None,
            OutcomeType.CI:                       None,
            OutcomeType.MCI:                      None,
            OutcomeType.DIABETES:                 DiabetesPrevalenceModelRepository(),
            OutcomeType.CHRONIC_KIDNEY_DISEASE:   ChronicKidneyDiseasePrevalenceModelRepository(),
            OutcomeType.CARDIOVASCULAR:           CVPrevalenceModelRepository(),
            OutcomeType.STROKE:                   StrokePrevalenceModelRepository(),
            OutcomeType.MI:                       MIPrevalenceModelRepository(),
            OutcomeType.NONCARDIOVASCULAR:        None,
            OutcomeType.DEMENTIA:                 DementiaPrevalenceModelRepository(),
            OutcomeType.EPILEPSY:                 EpilepsyPrevalenceModelRepository(),
            OutcomeType.DEATH:                    None,
            OutcomeType.QUALITYADJUSTED_LIFE_YEARS: None,
        }
        self.check_repository_completeness()

    def check_repository_completeness(self):
        for outcome in OutcomeType:
            if outcome not in list(self._repository.keys()):
                raise RuntimeError("OutcomePrevalenceModelRepository is incomplete")

    def has_prevalence_model(self, outcomeType):
        return self._repository.get(outcomeType) is not None
