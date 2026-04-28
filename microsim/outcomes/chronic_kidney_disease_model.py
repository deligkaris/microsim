from microsim.outcomes.outcome import Outcome, OutcomeType

class ChronicKidneyDiseaseModel:
    """Chronic kidney disease outcome model. First detection is gated on GFR < 60; once a CKD outcome
    has been recorded, a new outcome is emitted every wave thereafter regardless of current GFR."""

    def __init__(self):
        pass

    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.CHRONIC_KIDNEY_DISEASE, False)

    def get_next_outcome(self, person):
        if person.has_outcome(OutcomeType.CHRONIC_KIDNEY_DISEASE, inSim=False) or person._current_ckd:
            return self.generate_next_outcome(person)
        return None
