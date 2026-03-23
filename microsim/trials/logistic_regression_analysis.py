import sys
import warnings
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tools.sm_exceptions
from numpy.linalg import LinAlgError
from microsim.trials.regression_analysis import RegressionAnalysis

class LogisticRegressionAnalysis(RegressionAnalysis):
    def __init__(self):
        pass

    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        df = self.get_trial_outcome_df(trial, assessmentFunctionDict, assessmentAnalysis)
        blockFactors = trial.trialDescription.blockFactors
        formula = f"outcome ~ treatment"
        for blockFactor in blockFactors:
            formula += f" + {blockFactor}"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=statsmodels.tools.sm_exceptions.PerfectSeparationWarning)
                reg = smf.logit(formula, df).fit(disp=False)
            return reg.params['treatment'], reg.bse['treatment'], reg.pvalues['treatment'], reg.params['Intercept']
        except (LinAlgError, statsmodels.tools.sm_exceptions.PerfectSeparationWarning):
            print("Logistic regression failed (perfect separation/singular matrix), returning NaN.", file=sys.stderr)
            return np.nan, np.nan, np.nan, np.nan



