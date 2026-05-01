from scipy.special import expit

from microsim.outcomes.outcome import Outcome


class OutcomePrevalenceBase:
    """Base class for prevalence models that seed priorToSim outcomes when a Person is constructed.

    Subclasses MUST implement:
      - get_linear_predictor_for_person(person)
      - calc_linear_predictor_for_patient_characteristics(...)  (signature is subclass-specific)

    Subclasses set the class attribute `_outcomeType`. Subclasses whose Outcome carries
    phenotype data (e.g., StrokeOutcome) override `generate_prevalent_outcome`.
    """

    _outcomeType = None  # subclasses must set this

    def get_risk_for_person(self, person):
        return expit(self.get_linear_predictor_for_person(person))

    def get_linear_predictor_for_person(self, person):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_linear_predictor_for_person"
        )

    def calc_linear_predictor_for_patient_characteristics(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement calc_linear_predictor_for_patient_characteristics"
        )

    def generate_prevalent_outcome(self, person):
        return Outcome(self._outcomeType, fatal=False, priorToSim=True)

    def get_prevalent_outcome(self, person):
        if person._rng.uniform(size=1) < self.get_risk_for_person(person):
            return self.generate_prevalent_outcome(person)
        return None
