from mcm.outcome_model_type import OutcomeModelType
from mcm.gender import NHANESGender
from mcm.ascvd_outcome_model import ASCVDOutcomeModel
from mcm.statsmodel_cox_model import StatsModelCoxModel
from mcm.cox_regression_model import CoxRegressionModel
from mcm.cv_outcome_determination import CVOutcomeDetermination
from mcm.data_loader import load_model_spec

import numpy.random as npRand
import os
import json


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
        self._models[OutcomeModelType.CARDIOVASCULAR] = {
            "female": ASCVDOutcomeModel(
                age=0.106501, black_race=0.432440, sbp_x_sbp=0.000056, sbp=0.017666,
                bp_treatment=0.731678, diabetes=0.943970, current_smoker=1.009790,
                tot_chol_hdl_ratio=0.151318, age_x_black_race=-0.008580,
                sbp_x_treatment=-0.003647, sbp_x_black_race=0.006208,
                black_race_x_treatment=0.152968, age_x_sbp=-0.000153,
                black_race_x_diabetes=0.115232,
                black_race_x_current_smoker=-0.092231, black_race_x_tot_chol_hdl_ratio=0.070498,
                sbp_x_black_race_x_treatment=-0.000173, age_x_sbp_x_black_race=-0.000094,
                intercept=-12.823110
            ),
            "male": ASCVDOutcomeModel(
                age=0.064200, black_race=0.482835, sbp_x_sbp=-0.000061, sbp=0.038950,
                bp_treatment=2.055533, diabetes=0.842209, current_smoker=0.895589,
                tot_chol_hdl_ratio=0.193307, age_x_black_race=0,
                sbp_x_treatment=-0.014207, sbp_x_black_race=0.011609,
                black_race_x_treatment=-0.119460, age_x_sbp=0.000025,
                black_race_x_diabetes=-0.077214,
                black_race_x_current_smoker=-0.226771, black_race_x_tot_chol_hdl_ratio=-0.117749,
                sbp_x_black_race_x_treatment=0.004190, age_x_sbp_x_black_race=-0.000199,
                intercept=-11.679980
            ),

        }
        # This represents non-cardiovascular mortality..
        self._models[OutcomeModelType.NON_CV_MORTALITY] = self.initialize_cox_model(
            "nhanesMortalityModel")

    def get_risk_for_person(self, person, outcome, years=1):
        return self.select_model_for_person(person, outcome).get_risk_for_person(person, years)

    def select_model_for_person(self, person, outcome):
        models_for_outcome = self._models[outcome]
        if outcome == OutcomeModelType.NON_CV_MORTALITY:
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
