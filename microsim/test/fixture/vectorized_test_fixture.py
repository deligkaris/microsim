import unittest

import pandas as pd

from microsim.gcp_model import GCPModel
from microsim.alcohol_category import AlcoholCategory
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.population import Population
from microsim.person import Person
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus


class VectorizedTestFixture(unittest.TestCase):
    """
    Provides Pandas `DataFrame`s suitable for testing vectorized code.
    """

    _population_dataframe = None

    @classmethod
    def get_or_init_population_dataframe(cls):
        if VectorizedTestFixture._population_dataframe is None:
            test_person = Person(
                age=71,
                gender=NHANESGender.MALE,
                raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
                sbp=144.667,
                dbp=52.6667,
                a1c=9.5,
                hdl=34,
                totChol=191,
                bmi=30.05,
                ldl=110.0,
                trig=128,
                waist=45,
                anyPhysicalActivity=0,
                education=Education.COLLEGEGRADUATE,
                smokingStatus=SmokingStatus.FORMER,
                alcohol=AlcoholCategory.NONE,
                antiHypertensiveCount=0,
                statin=0,
                otherLipidLoweringMedicationCount=0,
                initializeAfib=(lambda _: None),
                randomEffects={'gcp': 0},
            )
            base_gcp = GCPModel().calc_linear_predictor(test_person)
            test_person._gcp.append([base_gcp])

            people = pd.Series([test_person])
            population = Population(people)
            VectorizedTestFixture._population_dataframe = population.get_people_current_state_and_summary_as_dataframe()
        return VectorizedTestFixture._population_dataframe

    def setUp(self):
        self.population_dataframe = VectorizedTestFixture.get_or_init_population_dataframe()
