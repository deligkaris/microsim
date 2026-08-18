from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel
from microsim.regression_models.linear_probability_risk_factor_model import LinearProbabilityRiskFactorModel
from microsim.regression_models.rounded_linear_risk_factor_model import RoundedLinearRiskFactorModel
from microsim.common.data_loader import load_regression_model

class RiskModelRepository:
    def __init__(self):
        self._repository = {}

    def get_model(self, name):
        return self._repository[name]

    def _initialize_linear_risk_model(self, referenceName, modelName, log=False):
        model = load_regression_model(modelName)
        self._repository[referenceName] = LinearRiskFactorModel(model, log)

    def _initialize_linear_probability_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = LinearProbabilityRiskFactorModel(model)

    def _initialize_int_rounded_linear_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = RoundedLinearRiskFactorModel(model)
