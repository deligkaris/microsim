import numpy as np
import pandas as pd

from microsim.alcohol_category import AlcoholCategory
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_model_repository import RiskModelRepository
from microsim.outcome import Outcome, OutcomeType
from microsim.person import Person
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.treatment import DefaultTreatmentsType, TreatmentStrategiesType
from microsim.stroke_outcome import StrokeOutcome

#useful to convert column names from the NHANES data to the names Microsim uses...
#Q: this probably belongs somewhere else...
microsimToNhanes = {DynamicRiskFactorsType.SBP.value: "meanSBP",
                    DynamicRiskFactorsType.DBP.value: "meanDBP",
                    DynamicRiskFactorsType.A1C.value: "a1c",
                    DynamicRiskFactorsType.HDL.value: "hdl",
                    DynamicRiskFactorsType.LDL.value: "ldl",
                    DynamicRiskFactorsType.TRIG.value: "trig",
                    DynamicRiskFactorsType.TOT_CHOL.value: "tot_chol",
                    DynamicRiskFactorsType.BMI.value: "bmi",
                    DynamicRiskFactorsType.WAIST.value: "waist",
                    DynamicRiskFactorsType.AGE.value: "age",
                    DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: 'anyPhysicalActivity',
                    DynamicRiskFactorsType.CREATININE.value: "serumCreatinine",
                    DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: "alcoholPerWeek",
                    DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: "antiHypertensive"}

class PersonFactory:
    """A class used to obtain Person-objects using data from a variety of sources."""

    @staticmethod
    def get_nhanes_person(x, initializationModelRepository):
        """Takes all Person-instance-related data via x and initializationModelRepository and organizes it,
           passes the organized data to the Person class and returns a Person instance."""

        rng = np.random.default_rng()

        name = x.name
   
        personStaticRiskFactors = {
                            StaticRiskFactorsType.RACE_ETHNICITY.value: NHANESRaceEthnicity(int(x.raceEthnicity)),
                            StaticRiskFactorsType.EDUCATION.value: Education(int(x.education)),
                            StaticRiskFactorsType.GENDER.value: NHANESGender(int(x.gender)),
                            StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus(int(x.smokingStatus))}
   
        #use this to get the bounds imposed on the risk factors in a bit
        rfRepository = RiskModelRepository()

        #TO DO: find a way to include everything here, including the rfs that need initialization
        #the PVD model would be easy to implement, eg with an estimate_next_risk_for_patient_characteristics function
        #but the AFIB model would be more difficult because it relies on statsmodel_logistic_risk file
        #for now include None, in order to create the risk factor lists correctly at the Person instance
        personDynamicRiskFactors = dict()
        for rfd in DynamicRiskFactorsType:
            if rfd==DynamicRiskFactorsType.ALCOHOL_PER_WEEK:
                personDynamicRiskFactors[rfd.value] = AlcoholCategory(x[rfd.value])
            else:
                if (rfd!=DynamicRiskFactorsType.PVD) & (rfd!=DynamicRiskFactorsType.AFIB):
                    personDynamicRiskFactors[rfd.value] = rfRepository.apply_bounds(rfd.value, x[rfd.value])
        personDynamicRiskFactors[DynamicRiskFactorsType.AFIB.value] = None
        personDynamicRiskFactors[DynamicRiskFactorsType.PVD.value] = None

        #Q: do we need otherLipid treatment? I am not bringing it to the Person objects for now.
        #A: it is ok to leave it out as we do not have a model to update this. It is also very rarely taking place in the population anyway.
        #also: used to have round(x.statin) but NHANES includes statin=2...
        personDefaultTreatments = {
                            DefaultTreatmentsType.STATIN.value: bool(x.statin),
                            #DefaultTreatmentsType.OTHER_LIPID_LOWERING_MEDICATION_COUNT.value: x.otherLipidLowering,
                            DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: x.antiHypertensiveCount}

        personTreatmentStrategies = dict(zip([strategy.value for strategy in TreatmentStrategiesType],
                                              #[None for strategy in range(len(TreatmentStrategiesType))]))
                                              [{"status": None} for strategy in range(len(TreatmentStrategiesType))]))

        personOutcomes = dict(zip([outcome for outcome in OutcomeType],
                                  [list() for outcome in range(len(OutcomeType))]))

        #If df originates from the NHANES df these columns will exist, but if drawing from the NHANES distributions, these will not be in the df
        if "selfReportStrokeAge" in x.index:
            #add pre-simulation stroke outcomes
            selfReportStrokeAge=x.selfReportStrokeAge
            #Q: we should not add the stroke outcome in case of "else"?
            if selfReportStrokeAge is not None and selfReportStrokeAge > 1:
                selfReportStrokeAge = selfReportStrokeAge if selfReportStrokeAge <= x.age else x.age
                personOutcomes[OutcomeType.STROKE].append((selfReportStrokeAge, StrokeOutcome(False, None, None, None, priorToSim=True)))
        if "selfReportMIAge" in x.index:
            #add pre-simulation mi outcomes
            selfReportMIAge=rng.integers(18, x.age) if x.selfReportMIAge == 99999 else x.selfReportMIAge
            if selfReportMIAge is not None and selfReportMIAge > 1:
                selfReportMIAge = selfReportMIAge if selfReportMIAge <= x.age else x.age
                personOutcomes[OutcomeType.MI].append((selfReportMIAge, Outcome(OutcomeType.MI, False, priorToSim=True)))

        person = Person(name,
                        personStaticRiskFactors,
                        personDynamicRiskFactors,
                        personDefaultTreatments,
                        personTreatmentStrategies,
                        personOutcomes)

        #TO DO: find a way to initialize these rfs above with everything else
        person._pvd = [initializationModelRepository[DynamicRiskFactorsType.PVD].estimate_next_risk(person)]
        person._afib = [initializationModelRepository[DynamicRiskFactorsType.AFIB].estimate_next_risk(person)]
        return person

