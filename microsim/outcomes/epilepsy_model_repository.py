from microsim.outcomes.epilepsy_model import EpilepsyIncidenceModel, EpilepsyPrevalenceModel

class EpilepsyModelRepository:
    def __init__(self):
        self._model = EpilepsyIncidenceModel()

    def select_outcome_model_for_person(self, person):
        return self._model

class EpilepsyPrevalenceModelRepository:
    def __init__(self):
        self._model = EpilepsyPrevalenceModel()

    def select_outcome_model_for_person(self, person):
        return self._model
