from microsim.outcomes.outcome import Outcome, OutcomeType

class DiabetesModel:
    """Diabetes outcome model. First detection is gated on A1C >= 6.5; once a diabetes outcome
    has been recorded, a new outcome is emitted every wave thereafter regardless of current A1C."""

    def __init__(self):
        pass

    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.DIABETES, False)

    def get_next_outcome(self, person):
        if person.has_outcome(OutcomeType.DIABETES, inSim=False) or person.has_diabetes():
            return self.generate_next_outcome(person)
        return None
