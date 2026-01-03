from microsim.outcomes.mci_model import MCIModel

class MCIModelRepository:
    def __init__(self):
        self._model = MCIModel()

    def select_outcome_model_for_person(self, person):
        return self._model 
