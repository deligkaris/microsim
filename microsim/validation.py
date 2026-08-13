import numpy as np
import pandas as pd

from microsim.population.population_factory import PopulationFactory
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial import Trial
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.trials.trial_outcome_assessor import AnalysisType
from microsim.trials.trial_type import TrialType
from microsim.outcomes.outcome import OutcomeType

class Burke2024:
    '''External validation standards for the NHANES arm of MICROSIM, from Burke et al.,
       "Development and validation of the Michigan Chronic Disease Simulation Model (MICROSIM)",
       PLOS ONE 19(5):e0300005 (2024).

       Every "standard" below is the external yardstick the paper validated MICROSIM against
       (published NHANES estimates, population-based incidence studies, US life tables, a
       meta-analysis of BP-lowering trials), NOT MICROSIM's own output. "paperValue" records what
       the paper's simulation obtained against that standard, and is where each tolerance comes
       from: no tolerance here is tighter than the deviation the paper itself reported and
       accepted. A check that fails is therefore a departure from Burke2024, not merely a
       departure from the literature.

       Not represented here, because the paper publishes them as figures with no numbers in the
       text: the probability of death by age against life tables (Fig 3) and dementia incidence
       against the Brookmeyer et al. meta-analysis (Fig 4). The "hypertension prevalence by the
       Eighth Joint National Committee criteria" row of Table 2 is also left out: JNC-8 uses
       age-dependent thresholds that the paper does not state, so any implementation of that row
       would be a guess at what was measured.'''

    #Table 2, the published survey-weighted NHANES 2007-2010 cohort.
    #The paper's simulation matched every one of these to within 0.1, so the tolerances are set by
    #what a 100,000-person weighted draw can be expected to reproduce rather than by the paper.
    baseline2007 = {
        "age, years (mean)": {"standard": 45.9, "tol": 1.0, "paperValue": 45.9},
        "female (%)": {"standard": 51.7, "tol": 1.5, "paperValue": 51.7},
        "white (%)": {"standard": 68.4, "tol": 2.0, "paperValue": 68.3},
        "black (%)": {"standard": 11.5, "tol": 2.0, "paperValue": 11.4},
        "hispanic (%)": {"standard": 13.6, "tol": 2.0, "paperValue": 13.6},
        "BMI, kg/m2 (mean)": {"standard": 28.5, "tol": 0.7, "paperValue": 28.5},
    }

    #Table 2, the published survey-weighted NHANES 2013 cohort with hypertension, defined there as
    #SBP > 140/90 mm Hg or taking any anti-hypertensive medication.
    baseline2013Hypertension = {
        #the paper's simulated population was 1.6 years younger than the NHANES one, its largest
        #Table 2 deviation, and the tolerance is set to accept it
        "age, years (mean)": {"standard": 60.0, "tol": 2.5, "paperValue": 58.4},
        "female (%)": {"standard": 50.0, "tol": 2.0, "paperValue": 49.7},
        "white (%)": {"standard": 71.0, "tol": 3.0, "paperValue": 69.6},
        "black (%)": {"standard": 14.0, "tol": 2.0, "paperValue": 14.0},
        "hispanic (%)": {"standard": 10.0, "tol": 2.0, "paperValue": 10.3},
        "BMI, kg/m2 (mean)": {"standard": 31.0, "tol": 0.7, "paperValue": 30.9},
        "SBP, mm Hg (mean)": {"standard": 133.4, "tol": 3.0, "paperValue": 132.1},
        "DBP, mm Hg (mean)": {"standard": 71.6, "tol": 2.0, "paperValue": 72.2},
        #Table 2 gives this row the same two values as the statin row underneath it, 41.0% for
        #NHANES and 41.4% for the simulation, which cannot be what the row measures: the table
        #defines this population as SBP > 140/90 mm Hg OR on an anti-hypertensive medication, and a
        #weighted NHANES 2013 draw of it comes out about 84% medicated here. The row is reported
        #rather than checked, since the number to check it against appears to be the statin one.
        "anti-hypertensive use (%)": {"status": "INFO", "paperValue": 41.4,
                                      "note": "Table 2 repeats the statin values in this row (41.0%/41.4%), "
                                              "which its own definition of the population rules out"},
        "statin use (%)": {"standard": 41.0, "tol": 2.0, "paperValue": 41.4},
    }

    #The two risk factors the paper quotes numbers for when describing Fig 2, for the NHANES 2017
    #pseudo-cohort it compares the advanced simulation against. Checking these says whether the
    #comparison population this code builds is the one the paper compared against.
    nhanes2017Comparator = {
        "DBP, mm Hg (mean)": {"standard": 71.6, "relTol": 0.05},
        "DBP, mm Hg (sd)": {"standard": 10.9, "relTol": 0.15},
        "totChol, mg/dL (mean)": {"standard": 196.8, "relTol": 0.05},
        "totChol, mg/dL (sd)": {"standard": 41.3, "relTol": 0.15},
    }

    #The vascular risk factors of Fig 2, whose distributions the paper reports the simulation
    #"generally closely reproduced" in both central tendency and variance after 18 years.
    fig2RiskFactors = [DynamicRiskFactorsType.SBP.value,
                       DynamicRiskFactorsType.DBP.value,
                       DynamicRiskFactorsType.A1C.value,
                       DynamicRiskFactorsType.HDL.value,
                       DynamicRiskFactorsType.LDL.value,
                       DynamicRiskFactorsType.TOT_CHOL.value,
                       DynamicRiskFactorsType.BMI.value]

    #Each Fig 2 risk factor is checked against the NHANES 2017 comparison population built in the
    #same run, so its standard is only known at run time. The four entries below are the ones the
    #paper puts a number on, including both deviations it documents as accepted; their tolerances
    #are what make those deviations pass. Every other risk factor uses fig2DefaultTolerance.
    fig2 = {
        #the paper's own over-prediction is +9.9%, so the tolerance is a little above that rather
        #than exactly at it, which would leave the value the paper reports sitting on the boundary
        (DynamicRiskFactorsType.DBP.value, "mean"): {"relTol": 0.12, "paperValue": 78.7,
                          "note": "Burke2024 reports 78.7 vs 71.6 mm Hg (+9.9%), its documented "
                                  "over-prediction of DBP"},
        (DynamicRiskFactorsType.DBP.value, "sd"): {"relTol": 0.28, "paperValue": 9.2,
                        "note": "Burke2024 reports SD 9.2 vs 10.9 (-15.6%)"},
        (DynamicRiskFactorsType.TOT_CHOL.value, "mean"): {"relTol": 0.10, "paperValue": 200.0,
                              "note": "Burke2024 reports 200.0 vs 196.8 mg/dL (+1.6%)"},
        (DynamicRiskFactorsType.TOT_CHOL.value, "sd"): {"relTol": 0.28, "paperValue": 31.0,
                            "note": "Burke2024 reports SD 31.0 vs 41.3 (-24.9%), its documented "
                                    "under-prediction of total cholesterol variance"},
    }
    fig2DefaultTolerance = {"mean": {"relTol": 0.10}, "sd": {"relTol": 0.28}}

    #Table 3 and the mortality sentence that follows it. Events per 100,000 population per year,
    #age-sex standardized. The stroke and MI standards published as a range are checked for
    #containment; the race-specific ones are single published estimates that the paper's own
    #simulation departed from substantially, and each tolerance is set to accept that departure.
    cvIncidence = {
        "MI incidence, all (per 100,000)": {"standard": (208., 284.), "paperValue": 234.,
                                            "note": "Kaiser Permanente 1999-2008"},
        "MI incidence, white (per 100,000)": {"standard": 199., "relTol": 0.30, "paperValue": 249.,
                                              "note": "Burke2024 obtained 249 vs 199 (+25%)"},
        "MI incidence, black (per 100,000)": {"standard": 189., "relTol": 0.30, "paperValue": 219.,
                                              "note": "Burke2024 obtained 219 vs 189 (+16%)"},
        "stroke incidence, all (per 100,000)": {"standard": (130., 400.), "paperValue": 153.,
                                                "note": "population-based studies 1999-2015"},
        "stroke incidence, white (per 100,000)": {"standard": 208., "relTol": 0.45, "paperValue": 123.,
                                                  "note": "Burke2024 obtained 123 vs 208 (-41%)"},
        "stroke incidence, black (per 100,000)": {"standard": 331., "relTol": 0.30, "paperValue": 243.,
                                                  "note": "Burke2024 obtained 243 vs 331 (-27%)"},
        #the disparity itself, which is what the paper claims to reproduce ("about double")
        "stroke incidence, black:white ratio": {"standard": (1.5, 2.5), "paperValue": 1.98, "fmt": "{:.2f}",
                                                "note": "the literature reports stroke incidence in Black "
                                                        "individuals at about double that in White individuals"},
        "mortality, all (per 100,000)": {"standard": 729., "relTol": 0.10, "paperValue": 699.,
                                         "note": "US population-level estimate"},
    }

    #Relative risks per added BP medication from the meta-analysis of BP-lowering trials that the
    #paper calibrated against, next to what its 15 simulations obtained.
    treatmentEffects = {
        "stroke RR, 1 BP medication added": {"standard": 0.79, "tol": 0.05, "paperValue": 0.76, "fmt": "{:.2f}"},
        "MI RR, 1 BP medication added": {"standard": 0.87, "tol": 0.05, "paperValue": 0.85, "fmt": "{:.2f}"},
    }

class Validation:

    @staticmethod
    def _percent_of(proportions, categoryValues):
        '''Returns, as a percentage, the share of a proportions dictionary (see
           Population.get_summary_at_index) held by the given category values. Keys are compared
           as ints, so the enum members the proportions are keyed by and plain ints both match.'''
        return 100. * sum([prop for key, prop in proportions.items() if int(key) in categoryValues])

    @staticmethod
    def _baseline_metrics(pop):
        '''Returns the quantities Burke2024 tabulates in Table 2 for a population at baseline, in
           the units the paper uses: means for the continuous variables, percentages for the
           categorical ones. Keyed by the metric names of the Burke2024 reference dictionaries.'''
        summary = pop.get_summary_at_index(0)
        continuous = summary["continuous"]
        proportions = summary["proportions"]
        #anti-hypertensives are a continuous treatment (a count), so the share of people taking any
        #of them is not in the proportions of the summary and is obtained from the counts directly
        antiHypertensiveCounts = pop.get_attr_at_index(DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value, 0)
        return {
            "age, years (mean)": continuous[DynamicRiskFactorsType.AGE.value]["mean"],
            "BMI, kg/m2 (mean)": continuous[DynamicRiskFactorsType.BMI.value]["mean"],
            "SBP, mm Hg (mean)": continuous[DynamicRiskFactorsType.SBP.value]["mean"],
            "DBP, mm Hg (mean)": continuous[DynamicRiskFactorsType.DBP.value]["mean"],
            "female (%)": Validation._percent_of(proportions[StaticRiskFactorsType.GENDER.value],
                                                 [NHANESGender.FEMALE]),
            "white (%)": Validation._percent_of(proportions[StaticRiskFactorsType.RACE_ETHNICITY.value],
                                                [RaceEthnicity.NON_HISPANIC_WHITE]),
            "black (%)": Validation._percent_of(proportions[StaticRiskFactorsType.RACE_ETHNICITY.value],
                                                [RaceEthnicity.NON_HISPANIC_BLACK]),
            "hispanic (%)": Validation._percent_of(proportions[StaticRiskFactorsType.RACE_ETHNICITY.value],
                                                   [RaceEthnicity.MEXICAN_AMERICAN, RaceEthnicity.OTHER_HISPANIC]),
            "anti-hypertensive use (%)": float(100. * np.mean([count>0 for count in antiHypertensiveCounts])),
            "statin use (%)": Validation._percent_of(proportions[DefaultTreatmentsType.STATIN.value], [1]),
        }

    @staticmethod
    def nhanes_baseline_pop():
        '''This function performs the simulation for the validation of the creation of the population (baseline models only).
           Returns {"2007": metrics, "2013Hypertension": metrics}, the Table 2 quantities of each of
           the two populations (see _baseline_metrics), for compare_baseline_pop_to_burke2024.'''
        print(f"\nVALIDATION OF BASELINE SIMULATED POPULATION")
        print("2007 Nhanes")
        popSize=100000
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=2007, personFilters=None, nhanesWeights=True, distributions=False)
        pop.print_baseline_summary()
        metrics2007 = Validation._baseline_metrics(pop)
        print("2013 Hypertension")
        #hypertension as Burke2024 defines it in Table 2: SBP > 140/90 mm Hg or on any
        #anti-hypertensive medication. Selecting only those on medication would make a population
        #that is 100% medicated, which is not the population the published estimates describe.
        pf = PersonFilterFactory.get_person_filter(["adult"])
        pf.add_filter(filterType="df",
                      filterName="hypertension",
                      filterFunction = lambda x: ((x[DynamicRiskFactorsType.SBP.value]>140) |
                                                  (x[DynamicRiskFactorsType.DBP.value]>90) |
                                                  (x[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]>0)))
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=2013, personFilters=pf, nhanesWeights=True, distributions=False)
        pop.print_baseline_summary()
        return {"2007": metrics2007, "2013Hypertension": Validation._baseline_metrics(pop)}

    @staticmethod
    def nhanes_over_time(nWorkers=5, path=None):
        '''Performs the over time validation of a population against the NHANES sample.
           The filters are used only for the NHANES comparison population from 2017.
           People that died prior to 2017 are not removed from the simulation population, if the simulation population is large enough
           and the death models work well, the resulting simulated population from an advancement of 18 years should be close to the
           NHANES comparison population.
           nWorkers determines the number of cores used
           path=None will result in displaying the figures whereas an actual path will export them to that path
           Returns {"simSummary", "nhanesSummary", "cvRates", "dementiaIncidence"}: the last-wave
           distribution summary of the advanced population and of the NHANES comparison population,
           the standardized CV rates and the dementia incidence, for compare_over_time_to_burke2024.'''
        nYears = 18
        popSize = 100000
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=1999, personFilters=None, nhanesWeights=True, distributions=False)
        pop.advance_parallel(nYears, None, nWorkers)
        pf = PersonFilterFactory.get_person_filter([])
        pf.add_filter(filterType="df",
                      filterName="lowAge",
                      filterFunction = lambda x: x[DynamicRiskFactorsType.AGE.value]>=36)
        pf.add_filter(filterType="df",
                      filterName="noImmigration",
                      filterFunction = lambda x: x["timeInUS"]>=4)
        nhanesPop = PopulationFactory.get_nhanes_population(n=popSize, year=2017, personFilters=pf, nhanesWeights=True, distributions=False)

        print("\nVALIDATION OF VASCULAR RISK FACTORS OVER TIME")
        pop.plot_vascular_rfs_last_wave(nhanesPop, path=path)
        pop.print_lastyear_summary_comparison(nhanesPop)
        print("\nVALIDATION OF CV EVENT INCIDENCE AND MORTALITY")
        cvRates = pop.get_cv_standardized_rates()
        pop.print_cv_standardized_rates(rates=cvRates)
        print("\nVALIDATION OF DEMENTIA INCIDENCE")
        dementiaIncidence = pop.get_outcome_incidence(OutcomeType.DEMENTIA)
        pop.print_outcome_incidence(outcomeType=OutcomeType.DEMENTIA, summary=dementiaIncidence)
        return {"simSummary": pop.get_summary_at_index(-1),
                "nhanesSummary": nhanesPop.get_summary_at_index(-1),
                "cvRates": cvRates,
                "dementiaIncidence": dementiaIncidence}

    @staticmethod
    def nhanes_treatment_effects(sampleSize=2000000, nWorkers=1):
        '''This function creates and advances a control and a treated population in order to estimate the
           BP medication treatment effect on the MI relative risk and the stroke relative risk.
           Returns {bpMedsAdded: {"strokeRR", "miRR", "strokeRRs", "miRRs"}}, the mean relative risk
           of each arm over its simulations and the individual simulation results, for
           compare_treatment_effects_to_burke2024.'''
        print("\nVALIDATION OF TREATMENT EFFECTS")
        nYears=5
        nSimulations = 4
        results = dict()
        #NHANES includes children, so the adult filter is what keeps this an adult treatment effect
        pf = PersonFilterFactory.get_person_filter(["adult"])
        for bpMedsAdded in [1,2,3,4]:
            miRRList = list()
            strokeRRList = list()
            print(f"\nbpMedsAdded={bpMedsAdded}")
            for i in range(nSimulations):
                td = NhanesTrialDescription(
                            trialType = TrialType.COMPLETELY_RANDOMIZED,
                            blockFactors=list(),
                            sampleSize = sampleSize,
                            duration = nYears,
                            treatmentStrategies = f"{bpMedsAdded}bpMedsAdded",
                            nWorkers = nWorkers,
                            personFilters=pf,
                            year=1999, nhanesWeights=True, distributions=False)
                toa = TrialOutcomeAssessorFactory.get_trial_outcome_assessor()
                tr = Trial(td)
                tr.run()
                tr.analyze(toa )
                strokeRR = tr.results[AnalysisType.RELATIVE_RISK.value]["strokeRR"][0]
                miRR = tr.results[AnalysisType.RELATIVE_RISK.value]["miRR"][0]
                miRRList += [miRR]
                strokeRRList += [strokeRR]
                print(f"\t\tsimulation={i}, strokeRR= {strokeRR:<8.2f}, miRR= {miRR:<8.2f}")
            print(f"    average of {nSimulations} simulations: strokeRR= {np.mean(strokeRRList):<8.2f}, miRR= {np.mean(miRRList):<8.2f}")
            print(f"         sd of {nSimulations} simulations: strokeRR= {np.std(strokeRRList):<8.2f}, miRR= {np.std(miRRList):<8.2f}")
            results[bpMedsAdded] = {"strokeRR": float(np.mean(strokeRRList)),
                                    "miRR": float(np.mean(miRRList)),
                                    "strokeRRs": strokeRRList,
                                    "miRRs": miRRList}
        return results

    # ==========================================================================
    # Comparison of the NHANES validation results with the Burke2024 standards
    # ==========================================================================

    @staticmethod
    def _check(name, value, standard=None, tol=None, relTol=None, paperValue=None, note=None,
               status=None, fmt="{:.1f}"):
        '''Compares one simulated quantity against its external standard in Burke2024 and returns
           the outcome as a dictionary, for _print_checks and _print_verdict.
           name: the metric, used as the key of the Burke2024 reference dictionaries as well
           value: what the simulation obtained
           standard: the published value, or a (low, high) tuple for a standard published as a range
           tol, relTol: the absolute/relative tolerance of a single-valued standard, one or the
                        other; a range standard is checked for containment and needs neither
           paperValue: what the simulation of Burke2024 obtained against the same standard
           status: pass "INFO" for a quantity that is reported but has no standard to judge it by
           Returns the display strings alongside the result, so that the formatting decisions of a
           metric (its units, its precision) are made once, here, where its reference data is.'''
        if status == "INFO":
            return {"name": name, "value": value, "standardString": "-", "deviationString": "-",
                    "toleranceString": "-", "paperValue": paperValue, "note": note,
                    "result": "INFO", "fmt": fmt}
        if type(standard) == tuple:
            low, high = standard
            result = "PASS" if (low <= value) & (value <= high) else "FAIL"
            standardString = f"{fmt.format(low)}-{fmt.format(high)}"
            outside = value - low if value < low else value - high
            deviationString = "in range" if result == "PASS" else f"{'+' if outside>0 else ''}{fmt.format(outside)}"
            toleranceString = "range"
        else:
            tolerance = tol if tol is not None else relTol*abs(standard)
            deviation = value - standard
            result = "PASS" if abs(deviation) <= tolerance else "FAIL"
            standardString = fmt.format(standard)
            deviationString = f"{'+' if deviation>=0 else ''}{fmt.format(deviation)}"
            toleranceString = f"+/-{fmt.format(tolerance)}"
        return {"name": name, "value": value, "standardString": standardString,
                "deviationString": deviationString, "toleranceString": toleranceString,
                "paperValue": paperValue, "note": note, "result": result, "fmt": fmt}

    @staticmethod
    def _check_against(references, name, value, **kwargs):
        '''Checks value against the Burke2024 reference data of metric name, see _check.
           An unknown name raises a KeyError: the metric names of the reference dictionaries are
           the only names the comparison functions may use.'''
        return Validation._check(name=name, value=value, **references[name], **kwargs)

    @staticmethod
    def _print_checks(title, checks):
        '''Prints one section of the comparison report: a row per metric with its standard, what
           the simulation obtained, the deviation and the tolerance it is allowed, what the
           simulation of Burke2024 obtained, and the result. Notes are printed underneath the rows
           that did not pass, where they explain what the tolerance was set from.'''
        print(f"\n{title}")
        print("-"*len(title))
        print(f"{'metric':<44}{'standard':>14}{'simulation':>12}{'deviation':>11}"
              f"{'tolerance':>11}{'Burke2024':>11}{'result':>8}")
        for check in checks:
            fmt = check["fmt"]
            paperValue = "-" if check["paperValue"] is None else fmt.format(check["paperValue"])
            print(f"{check['name']:<44}{check['standardString']:>14}{fmt.format(check['value']):>12}"
                  f"{check['deviationString']:>11}{check['toleranceString']:>11}{paperValue:>11}"
                  f"{check['result']:>8}")
            if (check["note"] is not None) & (check["result"] != "PASS"):
                print(f"{' '*4}note: {check['note']}")

    @staticmethod
    def _print_verdict(checks):
        '''Prints the overall verdict of the comparison and returns whether the simulation is
           consistent with every Burke2024 standard it was checked against. INFO quantities have no
           standard and so do not take part in the verdict.'''
        assessed = [check for check in checks if check["result"] != "INFO"]
        failed = [check for check in assessed if check["result"] == "FAIL"]
        informational = [check for check in checks if check["result"] == "INFO"]
        print("\n")
        if len(failed) == 0:
            print(f"VALIDATED: all {len(assessed)} checked quantities are consistent with the "
                  f"external standards of Burke2024.")
        else:
            print(f"NOT VALIDATED: {len(failed)} of {len(assessed)} checked quantities are "
                  f"inconsistent with the external standards of Burke2024:")
            for check in failed:
                fmt = check["fmt"]
                paperValue = "-" if check["paperValue"] is None else fmt.format(check["paperValue"])
                print(f"{' '*4}{check['name']}: standard {check['standardString']}, "
                      f"simulation {fmt.format(check['value'])}, allowed {check['toleranceString']}, "
                      f"Burke2024 obtained {paperValue}")
        print(f"{' '*4}Reported without a standard to check them against: {len(informational)}.")
        print(f"{' '*4}Dementia incidence is reported but not checked: Burke2024 publishes it as Fig 4 only,")
        print(f"{' '*4}with no numeric values in the text and no reference data for it in this repository.")
        return len(failed) == 0

    @staticmethod
    def compare_baseline_pop_to_burke2024(results):
        '''Compares the baseline populations against the published NHANES estimates of Table 2.
           results: what nhanes_baseline_pop returns.
           Returns a list of (section title, list of checks) for nhanes_burke2024_report.'''
        return [("Baseline population vs published NHANES 2007-2010 (Table 2)",
                 [Validation._check_against(Burke2024.baseline2007, name, results["2007"][name])
                  for name in Burke2024.baseline2007]),
                ("Baseline population with hypertension vs published NHANES 2013 (Table 2)",
                 [Validation._check_against(Burke2024.baseline2013Hypertension, name,
                                            results["2013Hypertension"][name])
                  for name in Burke2024.baseline2013Hypertension])]

    @staticmethod
    def compare_over_time_to_burke2024(results):
        '''Compares the population advanced for 18 years against the NHANES 2017 pseudo-cohort
           (Fig 2) and against the published event rates of Table 3.
           results: what nhanes_over_time returns.
           Returns a list of (section title, list of checks) for nhanes_burke2024_report.'''
        simSummary = results["simSummary"]["continuous"]
        nhanesSummary = results["nhanesSummary"]["continuous"]
        dbp = DynamicRiskFactorsType.DBP.value
        totChol = DynamicRiskFactorsType.TOT_CHOL.value

        #does the NHANES 2017 comparison population this code builds have the levels that Burke2024
        #reports for the population it compared the simulation against?
        comparatorValues = {"DBP, mm Hg (mean)": nhanesSummary[dbp]["mean"],
                            "DBP, mm Hg (sd)": nhanesSummary[dbp]["sd"],
                            "totChol, mg/dL (mean)": nhanesSummary[totChol]["mean"],
                            "totChol, mg/dL (sd)": nhanesSummary[totChol]["sd"]}
        comparatorChecks = [Validation._check_against(Burke2024.nhanes2017Comparator, name, value)
                            for name, value in comparatorValues.items()]

        #and does the simulation reproduce that population's vascular risk factor levels, which is
        #the claim Fig 2 makes? The standard of each of these is the comparison population itself.
        riskFactorChecks = []
        for riskFactor in Burke2024.fig2RiskFactors:
            for stat in ["mean", "sd"]:
                reference = Burke2024.fig2.get((riskFactor, stat),
                                               Burke2024.fig2DefaultTolerance[stat])
                riskFactorChecks += [Validation._check(name=f"{riskFactor} ({stat})",
                                                       value=simSummary[riskFactor][stat],
                                                       standard=nhanesSummary[riskFactor][stat],
                                                       **reference)]

        cvRates = results["cvRates"]
        miRates = cvRates[OutcomeType.MI]
        strokeRates = cvRates[OutcomeType.STROKE]
        cvChecks = [
            Validation._check_against(Burke2024.cvIncidence, "MI incidence, all (per 100,000)", miRates["all"]),
            Validation._check_against(Burke2024.cvIncidence, "MI incidence, white (per 100,000)", miRates["white"]),
            Validation._check_against(Burke2024.cvIncidence, "MI incidence, black (per 100,000)", miRates["black"]),
            Validation._check_against(Burke2024.cvIncidence, "stroke incidence, all (per 100,000)", strokeRates["all"]),
            Validation._check_against(Burke2024.cvIncidence, "stroke incidence, white (per 100,000)", strokeRates["white"]),
            Validation._check_against(Burke2024.cvIncidence, "stroke incidence, black (per 100,000)", strokeRates["black"]),
            Validation._check_against(Burke2024.cvIncidence, "mortality, all (per 100,000)",
                                      cvRates[OutcomeType.DEATH]["all"])]
        #the racial disparity in stroke incidence, which is the part of Table 3 the paper claims to
        #reproduce; a zero rate in white individuals would mean no events at all in that subgroup
        if strokeRates["white"] > 0:
            cvChecks += [Validation._check_against(Burke2024.cvIncidence, "stroke incidence, black:white ratio",
                                                   strokeRates["black"]/strokeRates["white"])]

        #dementia incidence has no numeric standard in Burke2024, it is published as Fig 4 only
        dementiaChecks = [Validation._check(name="dementia incidence, ages 65+ (per 100,000)",
                                            value=10**5 * results["dementiaIncidence"]["pooled_65_plus"],
                                            status="INFO",
                                            note="Burke2024 compares this with the Brookmeyer et al. "
                                                 "meta-analysis in Fig 4, without publishing the values")]
        return [("Vascular risk factors of the NHANES 2017 comparison population vs Burke2024 (Fig 2)", comparatorChecks),
                ("Simulated vascular risk factors after 18 years vs the NHANES 2017 comparison population (Fig 2)", riskFactorChecks),
                ("Event incidence and mortality vs population standards (Table 3)", cvChecks),
                ("Dementia incidence (Fig 4)", dementiaChecks)]

    @staticmethod
    def compare_treatment_effects_to_burke2024(results):
        '''Compares the estimated BP medication treatment effects against the meta-analysis of
           BP-lowering trials that Burke2024 calibrated against. That standard covers a single
           added medication, so the other arms are reported without being checked.
           results: what nhanes_treatment_effects returns.
           Returns a list of (section title, list of checks) for nhanes_burke2024_report.'''
        checks = []
        if 1 in results:
            checks += [Validation._check_against(Burke2024.treatmentEffects, "stroke RR, 1 BP medication added",
                                                 results[1]["strokeRR"]),
                       Validation._check_against(Burke2024.treatmentEffects, "MI RR, 1 BP medication added",
                                                 results[1]["miRR"])]
        for bpMedsAdded in sorted([arm for arm in results.keys() if arm != 1]):
            for outcome, key in [("stroke", "strokeRR"), ("MI", "miRR")]:
                checks += [Validation._check(name=f"{outcome} RR, {bpMedsAdded} BP medications added",
                                             value=results[bpMedsAdded][key],
                                             status="INFO", fmt="{:.2f}",
                                             note="Burke2024 validated a single added BP medication only")]
        return [("BP medication treatment effects vs trial meta-analysis", checks)]

    @staticmethod
    def nhanes_burke2024_report(baselineResults=None, overTimeResults=None,
                                treatmentEffectsResults=None, path=None, nWorkers=5):
        '''Compares the NHANES validation results against the external standards of Burke2024 and
           reports whether the simulation is still consistent with them.

           Each argument is what the corresponding nhanes_* function returns, and any that is None
           is obtained by running that function here. So

               Validation.nhanes_burke2024_report()

           runs the entire NHANES validation and reports on it, whereas passing results that are
           already in hand reports on them without running anything again - which is what the
           arguments are for, since these runs take hours.

           Returns {"checks": every check made, "validated": whether all of them passed}.'''
        if baselineResults is None:
            baselineResults = Validation.nhanes_baseline_pop()
        if overTimeResults is None:
            overTimeResults = Validation.nhanes_over_time(nWorkers=nWorkers, path=path)
        if treatmentEffectsResults is None:
            treatmentEffectsResults = Validation.nhanes_treatment_effects()

        print("\n\nCOMPARISON WITH BURKE2024")
        print("=========================")
        print("Burke et al., Development and validation of the Michigan Chronic Disease Simulation")
        print("Model (MICROSIM), PLOS ONE 19(5):e0300005, 2024.")
        print("Each quantity is checked against the external standard the paper validated against:")
        print("published NHANES estimates, population-based incidence studies, US population")
        print("mortality, and a meta-analysis of BP-lowering trials. The Burke2024 column is what")
        print("the simulation of the paper obtained against the same standard, and is where the")
        print("tolerances come from: no tolerance here is tighter than the deviation the paper")
        print("itself reported and accepted.")
        print("\nThe runs this reports on differ from the ones the paper describes: 100,000 people")
        print("rather than 500,000 at baseline and 250,000 over time, 4 simulations rather than 15")
        print("for the treatment effects, and a population advanced over time that, unlike the")
        print("paper's, is not restricted to people without prior stroke, MI or dementia.")

        sections = (Validation.compare_baseline_pop_to_burke2024(baselineResults) +
                    Validation.compare_over_time_to_burke2024(overTimeResults) +
                    Validation.compare_treatment_effects_to_burke2024(treatmentEffectsResults))
        checks = []
        for title, sectionChecks in sections:
            Validation._print_checks(title, sectionChecks)
            checks += sectionChecks
        return {"checks": checks, "validated": Validation._print_verdict(checks)}

    @staticmethod
    def nhanes(path=None, compare=True, nWorkers=5):
        '''Runs the entire NHANES validation.
           compare=True also compares the results with the external standards of Burke2024 and
           reports whether the simulation is consistent with them, see nhanes_burke2024_report.
           Returns the results of the three validation runs, and the comparison report when one
           was made.'''
        baselineResults = Validation.nhanes_baseline_pop()
        overTimeResults = Validation.nhanes_over_time(nWorkers=nWorkers, path=path)
        treatmentEffectsResults = Validation.nhanes_treatment_effects()
        results = {"baseline": baselineResults,
                   "overTime": overTimeResults,
                   "treatmentEffects": treatmentEffectsResults}
        if compare:
            results["report"] = Validation.nhanes_burke2024_report(baselineResults, overTimeResults,
                                                                   treatmentEffectsResults)
        return results

    @staticmethod
    def kaiser_baseline_pop(wmhSpecific=True):
        print(f"\nVALIDATION OF BASELINE SIMULATED POPULATION\n")
        popSize = 500000
        pop = PopulationFactory.get_kaiser_population(n=popSize, personFilters=None, wmhSpecific=wmhSpecific)
        pop.print_baseline_summary()
        pop.print_wmh_outcome_summary()
        print("\n")
        print(" "*25, "Reference for Kaiser population...")
        print(" "*16, "severity proportion")
        print(" "*25, "-"*20)
        print(f"{'no  0.707':>31}")
        print(f"{'mild  0.172':>31}")
        print(f"{'moderate  0.038':>31}")
        print(f"{'severe  0.015':>31}")
        print(f"{'unknown  0.069':>31}")
        print("\n")
        print(" "*21, "SBI proportion")
        print(" "*25, "-"*20)
        print(f"{'TRUE  0.044':>31}")

    @staticmethod
    def kaiser_over_time(wmhSpecific=True, nWorkers=1):
        print(f"\nVALIDATION OF SIMULATED POPULATION OVER TIME\n")
        print("Note: this function will return a dictionary of Pandas dataframes with the information needed to do a proportional hazards analysis...")
        print("Note: so ensure you will capture the return variable from this function call...")
        print("Note: because this might take a while...")
        popSize = 500000
        pop = PopulationFactory.get_kaiser_population(n=popSize, personFilters=None, wmhSpecific=wmhSpecific)
        pop.advance(11, nWorkers=nWorkers)
        groupStrings = {1:"CT SBI", 2: "CT WMD", 3: "CT BOTH", 0: "CT NONE", 5:"MRI SBI", 6:"MRI WMD", 7:"MRI BOTH", 4:"MRI NONE"}

        ratesRef = {"stroke": 12, "death": 27, "dementia": 11, "mi": 12}
        strokeRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.STROKE], wave=3)
        dementiaRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.DEMENTIA], wave=3)
        deathRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.DEATH], wave=3)
        miRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.MI], wave=3)
        rates = {"stroke": strokeRates, "dementia": dementiaRates, "death": deathRates, "mi": miRates}
        print(" "*12, "Printing outcome incidence rates at the end of year 4...")
        print(" "*12, "References: a Microsim simulation with all WMH-related models.\n")
        print(" "*12, "Outcome     Reference     Simulation")
        print(" "*12, "-"*40)
        for outcome in rates.keys():
            print(" "*10 + f"{outcome:>10}" + f"{ratesRef[outcome]:>14.1f} " + f"{rates[outcome]:>14.1f}")

        print("\n")
        print(" "*12, "Printing outcome incidence rates by SCD group and modality at the end of year 11...")
        print(" "*12, "References: Stroke-Kent2021, Wang2024, Mortality-Clancy2025, Dementia-Kent2022, MI-no available publication.\n")
        print(" "*12, "Mortality rates")
        print(" "*12, "-"*40)
        deathRatesRef = {1:61.5, 2: 63.8, 3: 84.9, 0:18.2, 5:49.2, 6:28.5, 7:53.7, 4:14.}
        deathMinCiRef = {1:59.1, 2:62.6,  3: 80.9, 0:17.8, 5:45.1, 6:27.6, 7:48.8, 4:13.4}
        deathMaxCiRef = {1:63.9, 2:65.1,  3:89.2,  0:18.5, 5:53.6, 6:29.4, 7:59.0, 4:14.6}
        deathRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.DEATH], wave=3)
        deathRatesList = list()
        print("     Group                  Reference     Simulation")
        for group in deathRatesRef.keys():
            deathRatesList += [ [f"{groupStrings[group]:>10} ", 
                                 f"{deathRatesRef[group]:>10.1f} ({deathMinCiRef[group]:>5.1f} - {deathMaxCiRef[group]:>4.1f} ) ",
                                 f"{deathRates[group]:>14.1f}"] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{deathRatesRef[group]:>10.1f} ({deathMinCiRef[group]:>5.1f} - {deathMaxCiRef[group]:>4.1f} ) " +
                  f"{deathRates[group]:>14.1f}")
        print("\n")
        print(" "*12, "Stroke rates")
        print(" "*12, "-"*40)
        strokeRatesRef = {1: 36.6, 2: 28.5, 3: 47.4, 0: 8.2, 5:31.2, 6: 13.,  7:34.5, 4: 4.8}
        strokeMinCiRef = {1: 34.9, 2: 27.7, 3: 44.5, 0: 8.,  5:28.,  6: 12.4, 7:30.6, 4: 4.5}
        strokeMaxCiRef = {1: 38.4, 2: 29.3, 3: 50.5, 0: 8.4, 5:34.6, 6: 13.6, 7:38.7, 4: 5.2}
        strokeRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.STROKE], wave=3)
        strokeRatesList = list()
        print("     Group                  Reference     Simulation")
        for group in strokeRatesRef.keys():
            strokeRatesList += [ [f"{groupStrings[group]:10} ", 
                                  f"{strokeRatesRef[group]:>4.1f} ({strokeMinCiRef[group]:>5.1f} - {strokeMaxCiRef[group]:>4.1f} ) ",
                                  f"{strokeRates[group]:<4.1f}" ] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{strokeRatesRef[group]:>10.1f} ({strokeMinCiRef[group]:>5.1f} - {strokeMaxCiRef[group]:>4.1f} ) " +
                  f"{strokeRates[group]:>14.1f}")
        print("\n")
        print(" "*12, "MI rates")
        print(" "*12, "-"*40)
        miRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.MI], wave=3)
        print("     Group                                Simulation")
        miRatesList = list()
        for group in groupStrings.keys():
            miRatesList += [ [f"{groupStrings[group]:>10} ",  
                              f"{miRates[group]:>14.1f}"] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{miRates[group]:>41.1f}")
        print("\n")
        print(" "*12, "Dementia rates")
        print(" "*12, "-"*40)
        dementiaRatesRef = {1:32.8, 2:37.7, 3:51.6, 0:6.7, 5:16.6, 6:9.6, 7:19.1, 4:2.9}
        dementiaMinCiRef = {1:31.,  2:36.7, 3:48.3, 0:6.5, 5:14.2, 6:9.1, 7:16.2, 4:2.7}
        dementiaMaxCiRef = {1:34.6, 2:38.7, 3:55.1, 0:6.9, 5:19.3, 6:10.1,7:22.4, 4:3.3}
        dementiaRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.DEMENTIA], wave=3)
        dementiaRatesList = list()
        print("     Group                  Reference     Simulation")
        for group in dementiaRatesRef.keys():
            dementiaRatesList += [ [f"{groupStrings[group]:>10} ", 
                                    f"{dementiaRatesRef[group]:>10.1f} ({dementiaMinCiRef[group]:>5.1f} - {dementiaMaxCiRef[group]:>4.1f} ) ",
                                    f"{dementiaRates[group]:>14.1f}"] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{dementiaRatesRef[group]:>10.1f} ({dementiaMinCiRef[group]:>5.1f} - {dementiaMaxCiRef[group]:>4.1f} ) " +
                  f"{dementiaRates[group]:>14.1f}")

        #obtain data for the stroke survival analysis, see figure 1 in Kent2021
        strokeInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.STROKE],
                                                   personFunctionsList = [lambda x: x.get_scd_group(), 
                                                                          lambda x: x.get_wmh_severity_by_modality_group()])
        strokeDf = pd.DataFrame(strokeInfo, columns=["time","event", "sbiwmhGroup", "severityGroup"])
  
        miInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.MI],
                                               personFunctionsList = [lambda x: x.get_scd_group(), 
                                                                      lambda x: x.get_wmh_severity_by_modality_group()])
        miDf = pd.DataFrame(miInfo, columns=["time","event", "sbiwmhGroup", "severityGroup"])

        #obtain data for the dementia survival analysis, see figure 2 in Kent2023
        dementiaInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.DEMENTIA],
                                                     personFunctionsList = [lambda x: x.get_wmh_severity_by_modality_group(),
                                                                            lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "sbi")),
                                                                            lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "wmh"))])
        dementiaDf = pd.DataFrame(dementiaInfo, columns=["time","event", "severityGroup", "sbi", "wmh"])

        deathInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.DEATH],
                                                  personFunctionsList = [lambda x: x.get_wmh_severity_by_modality_group(),
                                                                         lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "sbi")),
                                                                         lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "wmh"))])
        deathDf = pd.DataFrame(deathInfo, columns=["time","event", "severityGroup", "sbi", "wmh"])

        return {"death": deathDf, "mi": miDf, "stroke": strokeDf, "dementia": dementiaDf}

    @staticmethod
    def kaiser(wmhSpecific=True, nWorkers=1):
        Validation.kaiser_baseline_pop(wmhSpecific=wmhSpecific)
        dfs = Validation.kaiser_over_time(wmhSpecific=wmhSpecific, nWorkers=nWorkers)
        return dfs

 

