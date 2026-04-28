from microsim.outcomes.diabetes_model import DiabetesModel

class DiabetesModelRepository:
    def __init__(self):
        self._model = DiabetesModel()

    def select_outcome_model_for_person(self, person):
        return self._model
