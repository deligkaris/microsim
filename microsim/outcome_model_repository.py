from microsim.outcome_model_type import OutcomeModelType
from microsim.gender import NHANESGender
from microsim.ascvd_outcome_model import ASCVDOutcomeModel
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel
from microsim.cv_outcome_determination import CVOutcomeDetermination
from microsim.data_loader import load_model_spec
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.regression_model import RegressionModel
from microsim.gcp_model import GCPModel

import numpy.random as npRand


# This object is currently serving two purposes.
# First, it is (as its name implies), a repository fo routcome models
# Second, it is the gateway into logic around signing outcomes.
# To the extne that logic is complex, it has been moved into separate "Determination" classe
# e.g. CVOutcomeDetermination

class OutcomeModelRepository:

    def __init__(self):
        self.mi_case_fatality = CVOutcomeDetermination.default_mi_case_fatality
        self.secondary_mi_case_fatality = CVOutcomeDetermination.default_secondary_mi_case_fatality
        self.stroke_case_fatality = CVOutcomeDetermination.default_stroke_case_fatality
        self.secondary_stroke_case_fatality = CVOutcomeDetermination.default_secondary_stroke_case_fatality
        self.secondary_prevention_multiplier = CVOutcomeDetermination.default_secondary_prevention_multiplier

        # variable used in testing to control whether a patient will have a stroke or mi
        self.manualStrokeMIProbability = None

        self._models = {}
        femaleCVCoefficients = {'lagAge': 0.106501, 'black': 0.432440, 'lagSbp#lagSbp': 0.000056, 'lagSbp': 0.017666,
                                'current_bp_treatment': 0.731678, 'current_diabetes': 0.943970, 'current_smoker': 1.009790,
                                'lagAge#black': -0.008580, 'lagSbp#current_bp_treatment': -0.003647, 'lagSbp#black': 0.006208,
                                'black#current_bp_treatment': 0.152968, 'lagAge#lagSbp': -0.000153, 'black#current_diabetes': 0.115232,
                                'black#current_smoker': -0.092231, 'lagSbp#black#current_bp_treatment': -0.000173,
                                'lagAge#lagSbp#black': -0.000094, 'Intercept': -12.823110}

        maleCVCoefficients = {'lagAge': 0.064200, 'black': 0.482835, 'lagSbp#lagSbp': -0.000061, 'lagSbp': 0.038950,
                              'current_bp_treatment': 2.055533, 'current_diabetes': 0.842209, 'current_smoker': 0.895589,
                              'lagAge#black': 0, 'lagSbp#current_bp_treatment': -0.014207, 'lagSbp#black': 0.011609,
                              'black#current_bp_treatment': -0.119460, 'lagAge#lagSbp': 0.000025, 'black#current_diabetes': -0.077214,
                              'black#current_smoker': -0.226771, 'lagSbp#black#current_bp_treatment': 0.004190,
                              'lagAge#lagSbp#black': -0.000199, 'Intercept': -11.679980}

        self._models[OutcomeModelType.CARDIOVASCULAR] = {
            "female": ASCVDOutcomeModel(RegressionModel(coefficients=femaleCVCoefficients,
                                                        coefficient_standard_errors={key: 0 for key in femaleCVCoefficients},
                                                        residual_mean=0, residual_standard_deviation=0),
                                        tot_chol_hdl_ratio=0.151318, black_race_x_tot_chol_hdl_ratio=0.070498),

            "male": ASCVDOutcomeModel(RegressionModel(coefficients=maleCVCoefficients,
                                                      coefficient_standard_errors={key: 0 for key in maleCVCoefficients},
                                                      residual_mean=0, residual_standard_deviation=0),
                                      tot_chol_hdl_ratio=0.193307, black_race_x_tot_chol_hdl_ratio=-0.117749)

        }
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()

        # This represents non-cardiovascular mortality..
        self._models[OutcomeModelType.NON_CV_MORTALITY] = self.initialize_cox_model(
            "nhanesMortalityModel")

    def get_random_effects(self):
        return {'gcp': npRand.normal(0, 4.84)}

    def get_risk_for_person(self, person, outcome, years=1):
        return self.select_model_for_person(person, outcome).get_risk_for_person(person, years)

    def get_gcp(self, person):
        return self.get_risk_for_person(person, OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE)

    def select_model_for_person(self, person, outcome):
        models_for_outcome = self._models[outcome]
        if outcome == OutcomeModelType.NON_CV_MORTALITY or outcome == OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE:
            return models_for_outcome
        elif outcome == OutcomeModelType.CARDIOVASCULAR:
            gender_stem = "male" if person._gender == NHANESGender.MALE else "female"
            return models_for_outcome[gender_stem]

    def initialize_cox_model(self, modelName):
        model_spec = load_model_spec(modelName)
        return StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        outcomeDet = CVOutcomeDetermination(self.mi_case_fatality,
                                            self.stroke_case_fatality,
                                            self.secondary_mi_case_fatality,
                                            self.secondary_stroke_case_fatality,
                                            self.secondary_prevention_multiplier)
        return outcomeDet.assign_outcome_for_person(
            self, person, years, self.manualStrokeMIProbability)

    # Returns True if the model-based logic vs. the random comparison suggests death
    def assign_non_cv_mortality(self, person, years=1):
        riskForPerson = self.get_risk_for_person(person, OutcomeModelType.NON_CV_MORTALITY)
        if (npRand.uniform(size=1) < riskForPerson):
            return True
