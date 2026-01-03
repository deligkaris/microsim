from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType

class MCIModel:
    """Mild cognitive impairment model."""

    def __init__(self):
        pass

    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.MCI, False)

    def get_next_outcome(self, person):
        #return self.generate_next_outcome(person) if person.has_mci() else None   
        return self.generate_next_outcome(person) if self.get_mci_for_person(person) else None

    def get_mci_for_person(self, person):
        mci = person.has_mci()

        if mci:
            tst = TreatmentStrategiesType.WMD15.value
            if "wmd15MedsAdded" in person._treatmentStrategies[tst]:
                wmd15MedsAdded = person._treatmentStrategies[tst]['wmd15MedsAdded']
                mci = True if ((wmd15MedsAdded>0) and (person._rng.uniform(size=1)<0.8)) else False

            tst = TreatmentStrategiesType.WMD20.value
            if "wmd20MedsAdded" in person._treatmentStrategies[tst]:
                wmd20MedsAdded = person._treatmentStrategies[tst]['wmd20MedsAdded']
                mci = True if ((wmd20MedsAdded>0) and (person._rng.uniform(size=1)<0.75)) else False
        return mci
