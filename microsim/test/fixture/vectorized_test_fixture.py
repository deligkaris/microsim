import unittest

from microsim.outcomes.gcp_model import GCPModel
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.person import Person
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)


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
                raceEthnicity=RaceEthnicity.NON_HISPANIC_WHITE,
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
                creatinine=0.6,
                initializeAfib=(lambda _: None),
                randomEffects={"gcp": 0},
            )
            base_gcp = GCPModel().get_risk_for_person(test_person)
            test_person._gcp.append([base_gcp])

            VectorizedTestFixture._population_dataframe = init_vectorized_population_dataframe(
                [test_person]
            )
        return VectorizedTestFixture._population_dataframe

    def setUp(self):
        self.population_dataframe = VectorizedTestFixture.get_or_init_population_dataframe()
