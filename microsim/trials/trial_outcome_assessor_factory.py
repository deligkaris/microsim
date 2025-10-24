from microsim.trials.trial_outcome_assessor import TrialOutcomeAssessor, AnalysisType
from microsim.outcome import OutcomeType

class TrialOutcomeAssessorFactory:

    @staticmethod
    def get_trial_outcome_assessor(addCommonAssessments=True):
        '''This function adds some trial outcome assessments that are likely to be interesting from a trial.
        It also serves as an example of how trial outcome assessments can be added.
        To maintain output format quality keep the name of the assessments to 20 or less characters.'''
        toa = TrialOutcomeAssessor()
        if addCommonAssessments:
            toa.add_outcome_assessment("death", 
                                       {"outcome": lambda x: x.has_outcome(OutcomeType.DEATH)}, 
                                        AnalysisType.LOGISTIC.value)
            toa.add_outcome_assessment("anyEvent", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE,
                                                                  OutcomeType.DEMENTIA, OutcomeType.CI])}, 
                                        AnalysisType.LOGISTIC.value)
            toa.add_outcome_assessment("vascEventOrDeath", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE])}, 
                                        AnalysisType.LOGISTIC.value)
            toa.add_outcome_assessment("vascEvent", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.MI, OutcomeType.STROKE])}, 
                                        AnalysisType.LOGISTIC.value)
            toa.add_outcome_assessment("qalys", 
                                       {"outcome": lambda x: x.get_outcome_item_sum(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")}, 
                                        AnalysisType.LINEAR.value)
            toa.add_outcome_assessment("meanGCP", 
                                       {"outcome": lambda x: x.get_outcome_item_mean(OutcomeType.COGNITION, "gcp")}, 
                                        AnalysisType.LINEAR.value)
            toa.add_outcome_assessment("lastGCP", 
                                       {"outcome": lambda x: x.get_outcome_item_last(OutcomeType.COGNITION, "gcp")}, 
                                        AnalysisType.LINEAR.value)
            toa.add_outcome_assessment("cogEvent", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.CI, OutcomeType.DEMENTIA])}, 
                                        AnalysisType.LOGISTIC.value)
            toa.add_outcome_assessment("deathCox", 
                                       {"outcome": lambda x: x.has_outcome(OutcomeType.DEATH),
                                        "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.DEATH])},
                                        AnalysisType.COX.value)
            toa.add_outcome_assessment("cogEventCox", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.CI, OutcomeType.DEMENTIA]),
                                        "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.CI, OutcomeType.DEMENTIA])},
                                        AnalysisType.COX.value)
            toa.add_outcome_assessment("vascEventOrDeathCox",
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE]),
                                        "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE])},
                                        AnalysisType.COX.value)
            toa.add_outcome_assessment("strokeRR",
                                       {"outcome": lambda x: x.get_outcome_count(OutcomeType.STROKE)},
                                        AnalysisType.RELATIVE_RISK.value)
            toa.add_outcome_assessment("miRR",
                                       {"outcome": lambda x: x.get_outcome_count(OutcomeType.MI)},
                                        AnalysisType.RELATIVE_RISK.value)
            toa.add_outcome_assessment("cvRR",
                                       {"outcome": lambda x: x.get_outcome_count(OutcomeType.CARDIOVASCULAR)},
                                        AnalysisType.RELATIVE_RISK.value)
            toa.add_outcome_assessment("dementiaRR",
                                       {"outcome": lambda x: x.get_outcome_count(OutcomeType.DEMENTIA)},
                                        AnalysisType.RELATIVE_RISK.value)
            toa.add_outcome_assessment("ciRR",
                                       {"outcome": lambda x: x.get_outcome_count(OutcomeType.CI)},
                                        AnalysisType.RELATIVE_RISK.value)
            toa.add_outcome_assessment("dementiaOrCiRR",
                                       {"outcome": lambda x: x.get_any_outcome_count([OutcomeType.DEMENTIA, OutcomeType.CI])},
                                        AnalysisType.RELATIVE_RISK.value)
            toa.add_outcome_assessment("deathRR",
                                       {"outcome": lambda x: x.get_outcome_count(OutcomeType.DEATH)},
                                        AnalysisType.RELATIVE_RISK.value)
        return toa
