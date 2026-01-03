from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.dementia_model_repository import DementiaModelRepository
from microsim.outcomes.cognition_model_repository import CognitionModelRepository
from microsim.outcomes.qaly_model_repository import QALYModelRepository
from microsim.outcomes.cv_model_repository import CVModelRepository
from microsim.outcomes.stroke_partition_model_repository import StrokePartitionModelRepository
from microsim.outcomes.mi_partition_model_repository import MIPartitionModelRepository
from microsim.outcomes.non_cv_model_repository import NonCVModelRepository
from microsim.outcomes.death_model_repository import DeathModelRepository
from microsim.outcomes.ci_model_repository import CIModelRepository
from microsim.outcomes.mci_model_repository import MCIModelRepository
from microsim.outcomes.wmh_model_repository import WMHModelRepository
from microsim.outcomes.epilepsy_model_repository import EpilepsyModelRepository

class OutcomeModelRepository:
    """Holds the rules for all outcomes.
       Via a dictionary, this object selects the appropriate model repository for an outcome.
       The model repository will then select the appropriate model for a Person-instance (via a select_outcome_model_for_person function).
       The model then obtains the outcome for the Person-instance (via a get_next_outcome function).
       Outcomes are Outcome-instances when the only information we want is the occurence of the outcome, age, and fatality.
       Examples are death outcomes, mi outcomes.
       Outcomes are Outcome subclasses, eg StrokeOutcome, when more information about the outcome need to be stored, an outcome phenotype.
       Examples are StrokeOutcome (nihss, type etc), GCPOutcome (gcp), QALYOutcome (qaly)."""
    def __init__(self, wmhSpecific=True):
        self._wmhSpecific = wmhSpecific
        self._repository = {
                          OutcomeType.WMH: WMHModelRepository(),
                          OutcomeType.DEMENTIA: DementiaModelRepository(wmhSpecific = self._wmhSpecific),
                          OutcomeType.EPILEPSY: EpilepsyModelRepository(),
                          OutcomeType.COGNITION: CognitionModelRepository(),
                          OutcomeType.CI: CIModelRepository(),
                          OutcomeType.MCI: MCIModelRepository(),
                          OutcomeType.QUALITYADJUSTED_LIFE_YEARS: QALYModelRepository(),
                          OutcomeType.CARDIOVASCULAR: CVModelRepository(wmhSpecific = self._wmhSpecific),
                          OutcomeType.MI: MIPartitionModelRepository(),
                          OutcomeType.STROKE: StrokePartitionModelRepository(),
                          OutcomeType.NONCARDIOVASCULAR: NonCVModelRepository(wmhSpecific = self._wmhSpecific),
                          OutcomeType.DEATH: DeathModelRepository()}
        #must have a model repository for all outcome types
        self.check_repository_completeness()
 
    def check_repository_completeness(self):
        for outcome in OutcomeType:
            if outcome not in list(self._repository.keys()):
                raise RuntimeError("OutcomeModelRepository is incomplete")
