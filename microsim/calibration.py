import numpy as np
from scipy.special import logit, expit

from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.reference import Reference
from microsim.outcomes.cv_model import CVPrevalenceModel
from microsim.risk_factors.gender import NHANESGender

'''Analytic fits of prevalence model coefficients to the external reference rates in
   Reference.prevalence. Workflow: update Reference.prevalence, run the fit, bake the
   returned values into the model's _coefficients; test_calibration enforces agreement.
   Distinct from PopulationFactory.calibrate_prevalence, which empirically root-finds a
   riskScaling on a constructed population.'''


def _age_group_midpoint(group):
    '''"50-54" -> 52.0, the low bound plus 2, the midpoint convention of the fits.'''
    lo, hi = group.split("-")
    return float(lo) + 2.


def _fit_logit_linear(ratesByAgeGroup):
    '''OLS of logit(rate) on age-group midpoints.
       Returns {"Intercept": ..., "age": ...}, the shape the prevalence models bake.'''
    ages = np.array([_age_group_midpoint(group) for group in ratesByAgeGroup])
    rates = np.array(list(ratesByAgeGroup.values()))
    slope, intercept = np.polyfit(ages, logit(rates), 1)
    return {"Intercept": float(intercept), "age": float(slope)}


def fit_cv_prevalence():
    '''Fits CVPrevalenceModel._coefficients to the GBD CV rates in Reference.prevalence,
       per gender.'''
    reference = Reference.prevalence[OutcomeType.CARDIOVASCULAR.value]
    return {gender: _fit_logit_linear(reference[gender.name.lower()])
            for gender in NHANESGender}


def fit_stroke_prevalence(cvCoefficients=None):
    '''Fits StrokePrevalenceModel._coefficients to the GBD stroke rates in
       Reference.prevalence, per gender. The fitted quantity is the probability of stroke
       conditional on prevalent CV, so each GBD rate is divided by the CV model probability
       first: cvCoefficients defaults to CVPrevalenceModel._coefficients and must be
       refit before this whenever the CV rates change.'''
    if cvCoefficients is None:
        cvCoefficients = CVPrevalenceModel._coefficients
    reference = Reference.prevalence[OutcomeType.STROKE.value]
    coefficients = dict()
    for gender in NHANESGender:
        rates = reference[gender.name.lower()]
        cv = cvCoefficients[gender]
        conditionalRates = dict()
        for group, rate in rates.items():
            cvProbability = expit(cv["Intercept"] + cv["age"] * _age_group_midpoint(group))
            q = rate / cvProbability
            if q >= 1.:
                raise ValueError(
                    f"stroke rate {rate} in {gender.name.lower()} {group} exceeds the CV "
                    f"prevalence {cvProbability:.4f}; conditional probability is unfittable."
                )
            conditionalRates[group] = q
        coefficients[gender] = _fit_logit_linear(conditionalRates)
    return coefficients
