from microsim.outcomes.chronic_kidney_disease_model import ChronicKidneyDiseaseModel

class ChronicKidneyDiseaseModelRepository:
    def __init__(self):
        self._model = ChronicKidneyDiseaseModel()

    def select_outcome_model_for_person(self, person):
        return self._model
