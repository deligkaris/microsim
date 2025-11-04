from microsim.epilepsy_model import EpilepsyIncidenceModel

class EpilepsyModelRepository:
    def __init__(self):
        self._model = EpilepsyIncidenceModel()
        
    def select_outcome_model_for_person(self, person):
        return self._model
