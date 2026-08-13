import copy
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
from scipy.optimize import brentq
from enum import Enum
import math

from microsim.person.person_factory import PersonFactory
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.population.population import Population
from microsim.common.age_scope import AgeScope
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.population.population_model_repository import PopulationModelRepository, PopulationRepositoryType
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.risk_factors. cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository,
                                                                 CohortStaticRiskFactorModelRepository)
from microsim.default_treatments.default_treatment_model_repository import DefaultTreatmentModelRepository
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.population.standardized_population import StandardizedPopulation
from microsim.common.variable_type import VariableType
from microsim.outcomes.outcome import OutcomeType
from microsim.common.population_type import PopulationType
from microsim.common.data_loader import get_absolute_datafile_path
from microsim.risk_factors.modality import Modality

class VariablesUsedRecorder:
    """NOT IN USE at the moment: its only caller is apply_categorical_person_filters_on_groups, which
       stopped being used when get_nhanes_people moved to the construction the state populations use.

       Stands in for a dataframe row and remembers which variables were read from it.

       A person filter is a function, so nothing declares which variables it depends on. Handing it one
       of these instead of a row answers that question by observation: whatever the filter looks up ends
       up in variablesUsed, and the caller can then compare that against the categorical/continuous lists
       in PopulationFactory.variable_types.

       Only the variables the filter actually reaches for are recorded, so a filter that stops early
       records only what it read before stopping."""
    def __init__(self, row):
        object.__setattr__(self, "_row", row)
        object.__setattr__(self, "variablesUsed", set())

    def __getitem__(self, variable):
        self.variablesUsed.add(variable)
        return self._row[variable]

    def __getattr__(self, variable):
        #a filter that reaches for a variable as an attribute rather than a key is recorded the same way
        self.variablesUsed.add(variable)
        return getattr(self._row, variable)


class PopulationFactory:
    #the NHANES df as get_nhanesDf builds it, cached because every population build needs it
    _nhanesDf = None

    #the distributions the continuous variables are drawn from when distributions=True, cached because
    #neither the partition nor the fit depends on any argument: they are the same for the life of the
    #process and cost about 5 seconds to build
    _crudeDistributions = None

    #the mean of every continuous variable for every group of NHANES people, the mean a draw is shifted
    #to. Cached for the same reason: it takes no argument and every population build needs it.
    _groupMeans = None

    #NOT IN USE at the moment: only get_partitioned_nhanes_people reads it, and that is itself unused.
    #how many weighted draws get_partitioned_nhanes_people partitions in order to fit the distributions.
    #Measured on 1999: this yields ~1450 groups of which 644 hold at least 12 people, against 1558 groups
    #of which only 77 do when the whole unweighted year is used instead, at 39s against 20s. It does not
    #help with the singular covariance matrices, ~96% of groups have one at every n that was tried,
    #because the nine categorical variables cut a few thousand people into more cells than an eleven
    #dimensional covariance can be fit in.
    fitSampleSize = 40000

    nhanes_pop_attributes = {PopulationRepositoryType.STATIC_RISK_FACTORS.value:
                                                                    [StaticRiskFactorsType.GENDER.value,
                                                                     StaticRiskFactorsType.SMOKING_STATUS.value, 
                                                                     StaticRiskFactorsType.RACE_ETHNICITY.value,
                                                                     StaticRiskFactorsType.EDUCATION.value,
                                                                     StaticRiskFactorsType.MODALITY.value],
                             PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value: 
                                                                     [DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
                                                                      DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                                                                      DynamicRiskFactorsType.AGE.value, 
                                                                      DynamicRiskFactorsType.HDL.value, 
                                                                      DynamicRiskFactorsType.BMI.value, 
                                                                      DynamicRiskFactorsType.TOT_CHOL.value, 
                                                                      DynamicRiskFactorsType.TRIG.value, 
                                                                      DynamicRiskFactorsType.A1C.value, 
                                                                      DynamicRiskFactorsType.LDL.value, 
                                                                      DynamicRiskFactorsType.WAIST.value, 
                                                                      DynamicRiskFactorsType.CREATININE.value, 
                                                                      DynamicRiskFactorsType.SBP.value, 
                                                                      DynamicRiskFactorsType.DBP.value],
                             PopulationRepositoryType.DEFAULT_TREATMENTS.value: 
                                                                     [DefaultTreatmentsType.STATIN.value,
                                                                       DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value],
                             PopulationRepositoryType.OUTCOMES.value: 
                                                                      [OutcomeType.COGNITION.value,
                                                                       OutcomeType.CI.value,
                                                                       OutcomeType.CARDIOVASCULAR.value,
                                                                       OutcomeType.STROKE.value,
                                                                       OutcomeType.MI.value,
                                                                       OutcomeType.NONCARDIOVASCULAR.value,
                                                                       OutcomeType.DEMENTIA.value,
                                                                       OutcomeType.DEATH.value,
                                                                       OutcomeType.QUALITYADJUSTED_LIFE_YEARS.value]}
                                                  
    #these are used below to define groups ( = specific combinations of all NHANES categorical variables)
    # and to define which columns from the NHANES dataframe to model as Gaussians ( = all continuous variables present
    # in the original NHANES dataset).
    # The last point is important, the Gaussians model the continuous variables present in the original NHANES dataset
    # not all continuous variables present in the Microsim simulation (which includes more continuous variables not
    # present in the original NHANES dataset such as PVD)
    # The order of these two lists is important,as they define the column names of the final dataframe. The numpy arrays used in between do 
    # not keep track of which column is which attribute.
    nhanes_variable_types = {VariableType.CATEGORICAL.value:  [
                                                  StaticRiskFactorsType.MODALITY.value,
                                                  StaticRiskFactorsType.GENDER.value, 
                                                  StaticRiskFactorsType.SMOKING_STATUS.value, 
                                                  StaticRiskFactorsType.RACE_ETHNICITY.value, 
                                                  DefaultTreatmentsType.STATIN.value,
                                                  StaticRiskFactorsType.EDUCATION.value,
                                                  DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
                                                  DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                                                  DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value],
                             VariableType.CONTINUOUS.value:   [DynamicRiskFactorsType.AGE.value, 
                                                  DynamicRiskFactorsType.HDL.value, 
                                                  DynamicRiskFactorsType.BMI.value, 
                                                  DynamicRiskFactorsType.TOT_CHOL.value, 
                                                  DynamicRiskFactorsType.TRIG.value, 
                                                  DynamicRiskFactorsType.A1C.value, 
                                                  DynamicRiskFactorsType.LDL.value, 
                                                  DynamicRiskFactorsType.WAIST.value, 
                                                  DynamicRiskFactorsType.CREATININE.value, 
                                                  DynamicRiskFactorsType.SBP.value, 
                                                  DynamicRiskFactorsType.DBP.value]}
    #the order of the items in the two lists is critical because functions later on, eg draw from the distributions, depend on the order
    kaiser_variable_types = {VariableType.CATEGORICAL.value: [StaticRiskFactorsType.MODALITY.value,
                                                      StaticRiskFactorsType.GENDER.value, 
                                                      StaticRiskFactorsType.RACE_ETHNICITY.value, 
                                                      StaticRiskFactorsType.SMOKING_STATUS.value, 
                                                      DynamicRiskFactorsType.AFIB.value, 
                                                      DynamicRiskFactorsType.PVD.value, 
                                                      DefaultTreatmentsType.STATIN.value,
                                                      DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value],
                     VariableType.CONTINUOUS.value: [DynamicRiskFactorsType.AGE.value, 
                                                     DynamicRiskFactorsType.HDL.value, 
                                                     DynamicRiskFactorsType.A1C.value, 
                                                     DynamicRiskFactorsType.TOT_CHOL.value, 
                                                     DynamicRiskFactorsType.LDL.value, 
                                                     DynamicRiskFactorsType.TRIG.value, 
                                                     DynamicRiskFactorsType.CREATININE.value, 
                                                     DynamicRiskFactorsType.SBP.value, 
                                                     DynamicRiskFactorsType.DBP.value,
                                                     DynamicRiskFactorsType.BMI.value, 
                                                     DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]}

    @staticmethod
    def variable_types(varType=VariableType.CATEGORICAL.value, popType=PopulationType.NHANES.value):
        if popType==PopulationType.NHANES.value:
            return PopulationFactory.nhanes_variable_types[varType]
        elif popType==PopulationType.KAISER.value:
            return PopulationFactory.kaiser_variable_types[varType]
        else:
            raise RuntimeError("Unrecognized population type in PopulationFactory.variable_types.")       

    @staticmethod
    def get_pop_attributes(popType=PopulationType.NHANES.value):
        if popType == PopulationType.NHANES.value:
            return PopulationFactory.nhanes_pop_attributes 
        elif popType == PopulationType.KAISER.value:
            return PopulationFactory.kaiser_pop_attributes
        else:
            raise RuntimeError("Population type not a valid one in PopulationFactory.get_pop_attributes.")      

    @staticmethod
    def get_population(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population(**kwargs)
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_population(**kwargs)
        elif popType == PopulationType.STATE:
            return PopulationFactory.get_state_population(**kwargs)
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population function.")

    @staticmethod
    def get_people(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_people(**kwargs)
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_people(**kwargs)
        elif popType == PopulationType.STATE:
            return PopulationFactory.get_state_people(**kwargs)
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_people function.")

    @staticmethod
    def get_population_model_repo(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population_model_repo()
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_population_model_repo(**kwargs)
        elif popType == PopulationType.STATE:
            return PopulationFactory.get_nhanes_population_model_repo()
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population_model_repo function.")

    @staticmethod
    def set_index_in_people(people, start=0):
        """Once people are created, its Person-objects do not have a unique index.
           This function assigns a unique index to every Person-object in people."""
        list(map(lambda person, i: setattr(person, "_index", i+start), people, range(people.shape[0])))

    @staticmethod
    def get_nhanesDf():
        """Reads and modifies the NHANES dataframe so that it is ready to be used in the simulation.
           Returns a Pandas df with the NHANES information as exists in Microsim.
           The df is built once and kept in _nhanesDf: reading the file and converting the columns
           costs about 14 seconds, and every population build needs the df. Callers get a copy of
           the cached df, some of them add columns to what they get back (see get_treatment_weights
           and get_partitioned_nhanes_people_crude), which would corrupt the cache for everyone else."""
        if PopulationFactory._nhanesDf is None:
            nhanesDf = pd.read_stata(get_absolute_datafile_path("fullyImputedDataset.dta"))
            #in Person-objects, the attribute name is used
            #the column holding the row number is called index in the data file, renaming a level_0
            #column (which pandas produces only when reset_index runs on an index already named index)
            #silently did nothing, so no name column existed and the distributions path could not merge on it
            nhanesDf = nhanesDf.rename(columns={"index":"name"})
            #rename the columns that have different column names than the ones that appear in Microsim
            nhanesDf = PopulationFactory.rename_df_columns(nhanesDf, PersonFactory.microsimToNhanes)
            #convert the integers to booleans because in the simulation we always use bool for these
            for col in [DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value, DefaultTreatmentsType.STATIN.value]:
                nhanesDf[col] = nhanesDf[col].astype(bool)
            #convert drinks per week to category
            nhanesDf[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value] = nhanesDf.apply(lambda x:
                                                                                     AlcoholCategory.get_category_for_consumption(x[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]), axis=1)
            #convert these columns to int type
            for col in [StaticRiskFactorsType.RACE_ETHNICITY.value,
                        StaticRiskFactorsType.EDUCATION.value,
                        StaticRiskFactorsType.GENDER.value,
                        StaticRiskFactorsType.SMOKING_STATUS.value]:
                nhanesDf[col] = nhanesDf[col].astype(int)
            PopulationFactory._nhanesDf = nhanesDf
        return PopulationFactory._nhanesDf.copy()

    @staticmethod
    def get_kaiserDf(csvFile):
        """Reads and modifies the Kaiser file so that it is ready to be used in the simulation.
           Returns a Pandas df with the Kaiser information as named in Microsim."""
        df = pd.read_csv(csvFile).dropna()
        #TO DO: needs to be FIXED, or REMOVED
        #df = df.loc[ (df["AHL_nonStatin"]==0) ]
        #df = df.drop("AHL_nonStatin", axis=1)
        #if 'weight' in df.columns:
        #    df = df.drop('weight', axis=1)
        df = PopulationFactory.rename_df_columns(df, PersonFactory.microsimToKaiser)
        df = df.astype({StaticRiskFactorsType.SMOKING_STATUS.value: 'int',
                        DynamicRiskFactorsType.AFIB.value:'bool',
                        DynamicRiskFactorsType.PVD.value:'bool',
                        DefaultTreatmentsType.STATIN.value:'int', 
                        DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value:'bool',
                        #"age":'int'}).reset_index()
                       }).reset_index()
        #map, not replace: replacing strings with integers makes pandas downcast the object column
        #silently, which is deprecated and warns. map does the same substitution without the downcast.
        #The astype below still catches a value missing from the dict, as map turns it into NaN.
        df[StaticRiskFactorsType.GENDER.value] = df[StaticRiskFactorsType.GENDER.value].map({'F': 2, 'M': 1}).astype('int')
        df[StaticRiskFactorsType.RACE_ETHNICITY.value] = df[StaticRiskFactorsType.RACE_ETHNICITY.value].map(
                                        {'Black': RaceEthnicity.NON_HISPANIC_BLACK.value,
                                        'Asian and Pacific Islander': RaceEthnicity.ASIAN.value,
                                        'White': RaceEthnicity.NON_HISPANIC_WHITE.value,
                                        'Multiple/Other/Unknown': RaceEthnicity.OTHER.value,
                                        'Hispanic': RaceEthnicity.OTHER_HISPANIC.value}).astype('int')
        df[StaticRiskFactorsType.MODALITY.value] = df[StaticRiskFactorsType.MODALITY.value].replace({"CT": Modality.CT.value,
                                                                                                     "MR": Modality.MR.value})
        return df

    @staticmethod
    def rename_df_columns(df, microsimToDfDict):
        '''Dataframes that we typically use to import person data, eg NHANES, have column names that are different than microsim attributes.
        This function takes a dictionary that helps convert those column names to the exact names that microsim uses.'''
        for key, value in microsimToDfDict.items():
            if key!=value:
                df = df.rename(columns={value:key})
        return df

    @staticmethod
    def check_nhanes_people_arguments(n=None, year=None, nhanesWeights=False, distributions=False, customWeights=None, maxDraws=None):
        '''Raises RuntimeError for any argument of get_nhanes_people that cannot be honored, see its
           docstring for what each one means. Called before any of the work, so a call that cannot be
           served fails in 0.000s rather than after the distributions have been fit and every row redrawn.
           The order of the checks is what decides which message a call gets, see the two comments below.'''

        if (year is not None) & (year not in [2011, 2015, 2007, 2003, 2009, 2001, 2005, 1999, 2013, 2017]):
            raise RuntimeError(f"NHANES data for year {year} is not available")

        #n is a count of Person-objects: a float is not rounded here, since which way to round is the
        #caller's decision, and n=0 is refused because an empty population advances without error and
        #reports a prevalence of 0 for every outcome, so it reads as a result rather than as a mistake.
        #bool is excluded explicitly, n=True is far more likely a flag typed into the wrong argument.
        #These two use 'and', not the '&' below, because '&' evaluates both sides and both right-hand
        #sides raise on n=None
        if (n is not None) and (isinstance(n, bool) or not isinstance(n, (int, np.integer))):
            raise RuntimeError(f"""Cannot build a population of n={n!r} people: n is a count of
                                   Person-objects and must be an integer, not a {type(n).__name__}.""")

        if (n is not None) and (n < 1):
            raise RuntimeError(f"""Cannot build a population of n={n} people: n is the number of
                                   Person-objects to return and must be at least 1. Pass n=None to get
                                   every NHANES person of that year instead.""")

        #this one comes before the nhanesWeights checks, which would otherwise report this combination
        #as the both-weight-kinds one and give the less useful of the two messages
        if distributions & (customWeights is not None):
            raise RuntimeError("""Cannot use customWeights with distributions=True. With distributions the
                                  categorical variables and the age of each person still come from the
                                  NHANES rows and only the continuous variables are drawn, so it is the
                                  sampling of those rows that decides who the population is made of, and
                                  nhanesWeights is what makes that sampling representative.""")

        #the type is checked because the checks below combine nhanesWeights with '&': a non-bool would
        #either raise a TypeError out of the operator or, worse, pass through it silently. numpy bools
        #are accepted for the same reason numpy integers are, they arrive from dataframe comparisons
        if not isinstance(nhanesWeights, (bool, np.bool_)):
            raise RuntimeError(f"""nhanesWeights must be True or False, not {nhanesWeights!r}
                                   ({type(nhanesWeights).__name__}). It says whether the NHANES rows are
                                   sampled with the survey weights; it is not inferred from any other
                                   argument, and None is not a way of leaving it open.""")

        if nhanesWeights & (customWeights is not None):
            raise RuntimeError("Cannot use both nhanesWeights (nhanesWeights=True) and custom weights (customWeights is not None).")

        if nhanesWeights & (n is None):
            raise RuntimeError("""Cannot set nhanesWeights True without specifying n.
                                  NHANES weights are defined for each year independently and for sampling
                                  to occur the sampling size is needed.""")

        #WTINT2YR weighs a person only against the others of their own two-year cycle, so sampling every
        #year at once with these weights builds a population nobody can interpret
        if nhanesWeights & (year is None):
            raise RuntimeError("""Cannot use nhanesWeights True with year=None. The NHANES weights are
                                  defined for each survey year independently, so they cannot weigh people
                                  of different years against each other. Pass a year, or nhanesWeights
                                  False, or supply customWeights built for a pooled draw.""")

        if (customWeights is not None) & (n is None):
            raise RuntimeError("Cannot use customWeights without specifying n, for sampling to occur the sampling size is needed.")

        if (maxDraws is not None) & (n is None):
            raise RuntimeError("""Cannot use maxDraws without specifying n. maxDraws budgets the sampling
                                  of NHANES rows and without an n no sampling takes place at all.""")

    @staticmethod
    def get_nhanes_people(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None, outcomePrevalenceModelRepository=None, maxDraws=None):
        '''Returns a Pandas Series of Person-objects built from NHANES, with or without sampling.
           year: NHANES survey year, 1999 to 2017 in steps of 2. year=None uses every year at once, and is
              refused with nhanesWeights=True since those weights cannot weigh a 1999 person against a 2017
              one; a weighted pooled draw needs customWeights.
           n: number of Person-objects to return, honored in every sampling mode. n=None means no sampling:
              every NHANES person of that year that passes the filters is returned. nhanesWeights and
              customWeights both require an n.
           nhanesWeights: sample the rows with the survey weights (WTINT2YR), which is what makes a sample
              representative of the US population of that year. Must be True or False, defaults to False and
              is inferred from nothing, so the call itself says whether a population is weighted. Requires
              an n; refused with customWeights and with year=None.
           distributions: keep the categorical variables and the age of each row and redraw only the
              continuous ones from Gaussians fit to NHANES, so which rows are sampled still decides the
              make-up of the population. The redraw happens per sampled row, so a row drawn twice yields
              two people with different continuous variables. The df-level filters move after the sampling
              with it, since they have to hold for the drawn values. Refused with customWeights.
           maxDraws: budget on the NHANES rows sampled to reach n people, default max(100*n, 500) in
              bring_people_to_target_n. Filters reject a row only after it is drawn, so a selective set
              needs many more than n draws (0.3% acceptance is ~330 draws per person against a budget of
              100) and raises asking for this argument. Requires an n.
           Sampling is always with replacement and returns exactly n people: filters run after a row is
           drawn and whatever they drop is drawn again with the same weights (see bring_people_to_target_n).
           Without distributions the df-level filters run before the sampling, for speed and memory, which
           does not affect the relative make-up of the people returned.
           Which combinations of the arguments are refused, and why, is in check_nhanes_people_arguments.'''

        PopulationFactory.check_nhanes_people_arguments(n=n, year=year, nhanesWeights=nhanesWeights, distributions=distributions, customWeights=customWeights, maxDraws=maxDraws)

        nhanesDf = PopulationFactory.get_nhanesDf()

        if year is not None: #if year is None, then use the entire dataframe
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]

        if personFilters is None: #since we started including children in the NHANES df, by default use an adult filter on the df
            personFilters = PersonFilterFactory.get_person_filter(["adult"])
        elif "adult" not in personFilters.filters["df"]: #warn only when the caller's filters do not already exclude children
            print("Warning: NHANES populations now include children by default. Add an age filter for adults only.")

        #with distributions the continuous variables of a row are replaced by a draw, exactly as the state
        #populations are built: from a Gaussian fit on gender, race ethnicity, education and age, shifted
        #to the mean of the group the row belongs to (see group_key_frame). Grouping on those four only,
        #over all years and with overlapping age windows, is what leaves enough people per group to fit a
        #covariance matrix that is not singular
        crudeDistributions = PopulationFactory.get_crude_distributions() if distributions else None

        #without distributions the df-level filters run here, before the sampling, so rows that cannot
        #become people are not carried through it. With distributions they have to hold for the drawn
        #values, which do not exist until a row is sampled, so they run on each drawn row instead
        if not distributions:
            nhanesDf = PopulationFactory.apply_person_filters_on_df(personFilters, nhanesDf)
            if nhanesDf.shape[0] == 0:
                raise RuntimeError("""The df-level filters of personFilters rejected every row, so there is
                                      nobody left to build Person-objects from.""")

        #the weights are picked here, and not with the checks above, because the NHANES weights have to
        #come from the df the sampling is done on, ie after the filters have run
        if nhanesWeights:
            weights = nhanesDf.WTINT2YR
        elif customWeights is not None:
            weights = customWeights
        else:
            weights = None

        imr = InitializationModelRepository()
        if n is None:
            #no sampling: every row becomes a person, so with distributions every row is redrawn exactly
            #once and there is no row for a second draw to differ from
            nhanesDfForPeople = nhanesDf
            if distributions:
                nhanesDfForPeople = PopulationFactory.redraw_continuous_variables(nhanesDfForPeople, crudeDistributions)
                nhanesDfForPeople = PopulationFactory.apply_person_filters_on_df(personFilters, nhanesDfForPeople)
                if nhanesDfForPeople.shape[0] == 0:
                    raise RuntimeError("""The df-level filters of personFilters rejected every drawn row, so
                                          there is nobody left to build Person-objects from.""")
            people = pd.DataFrame.apply(nhanesDfForPeople, PersonFactory.get_person, popType=PopulationType.NHANES.value, initializationModelRepository=imr, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, axis="columns")
            people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
            #the third way to end up with nobody, guarded like the two df-level ones above it: an empty
            #population advances without error and reports a prevalence of 0 for every outcome, so an
            #unguarded wipe-out produces numbers that look like results. The n path cannot reach here,
            #there the same filters exhaust the draw budget and raise with their acceptance rate
            if people.shape[0] == 0:
                raise RuntimeError(f"""The person-level filters of personFilters
                                       ({sorted(personFilters.filters["person"].keys())}) rejected all
                                       {nhanesDfForPeople.shape[0]} of the Person-objects built, so the
                                       population would be empty.""")
        else:
            #the draw, the redraw and both levels of filtering all happen in the one loop, which keeps
            #drawing until n people have passed: filters drop people only after a row has been drawn
            people = PopulationFactory.bring_people_to_target_n(n, pd.Series([], dtype=object), nhanesDf, personFilters, popType=PopulationType.NHANES.value, initializationModelRepository=imr, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, weights=weights, maxDraws=maxDraws, distributions=crudeDistributions)

        #the labels of the returned Series are an artifact of how the people were assembled and mean
        #different things per branch: NHANES row numbers when nothing is sampled, repeated labels once a
        #row is drawn more than once, a per-batch renumbering from 0 once apply_person_filters_on_people
        #rebuilds each batch. They are dropped rather than handed out as if they identified anything:
        #what identifies a person is _index, set right below, and _name is the NHANES row they came from
        people = people.reset_index(drop=True)
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_nhanes_population_model_repo(riskScaling=None):
        """Return the default, self-consistent set of models for advancing an NHANES Population."""
        return PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                         DefaultTreatmentModelRepository(),
                                         OutcomeModelRepository(riskScaling=riskScaling),
                                         CohortStaticRiskFactorModelRepository())

    @staticmethod
    def get_kaiser_population_model_repo(wmhSpecific=True, riskScaling=None):
        """Return the default, self-consistent set of models for advancing a Kaiser Population."""
        return PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                         DefaultTreatmentModelRepository(),
                                         OutcomeModelRepository(wmhSpecific=wmhSpecific, riskScaling=riskScaling),
                                         CohortStaticRiskFactorModelRepository())

    @staticmethod
    def get_nhanes_population(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None, riskScaling=None, prevalenceRiskScaling=None, maxDraws=None):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.
           Person attributes can originate either from the NHANES dataset directly or from distributions fit to the NHANES dataset.
           nhanesWeights: True or False, defaulting to False and inferred from nothing, see get_nhanes_people.
           riskScaling: optional dict[OutcomeType, float] applied to per-outcome risk inside the OutcomeModelRepository.
           prevalenceRiskScaling: optional dict[OutcomeType, float] applied to per-outcome priorToSim risk inside the OutcomePrevalenceModelRepository.
           maxDraws: budget on the NHANES rows sampled to reach n people, raise it for restrictive
              personFilters, see get_nhanes_people.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, personFilters=personFilters, nhanesWeights=nhanesWeights, distributions=distributions, customWeights=customWeights, outcomePrevalenceModelRepository=OutcomePrevalenceModelRepository(riskScaling=prevalenceRiskScaling), maxDraws=maxDraws)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo(riskScaling=riskScaling)
        return Population(people, popModelRepository)

    @staticmethod
    def get_kaiser_population(n=1000, personFilters=None, wmhSpecific=True, riskScaling=None):
        people = PopulationFactory.get_kaiser_people(n=n, personFilters=personFilters)
        popModelRepository = PopulationFactory.get_kaiser_population_model_repo(wmhSpecific=wmhSpecific, riskScaling=riskScaling)
        return Population(people, popModelRepository)

    @staticmethod
    def calibrate_prevalence(scaleOutcomeType, targetOutcomeType, target, scope,
                             popType, peopleArgs, baselineRiskScaling=None):
        '''Empirically find the OutcomePrevalenceModelRepository riskScaling on
           `scaleOutcomeType` such that the realized priorToSim prevalence of
           `targetOutcomeType` in `scope` equals `target`.

           When `scaleOutcomeType == targetOutcomeType` this calibrates a single outcome
           directly. When they differ, the scale outcome must precede the target in
           OutcomeType seeding order so its priorToSim status can cascade into the
           target's linear predictor — the canonical case is stroke/MI, whose prevalence
           models short-circuit unless CV was seeded as priorToSim.

           scaleOutcomeType: outcome whose prevalence riskScaling is the unknown. Its
                             prevalence model must honor riskScaling; MI and COGNITION
                             do not and are refused.
           targetOutcomeType: outcome whose realized priorToSim prevalence the search
                              drives to `target`. May equal scaleOutcomeType.
           target:           desired realized prevalence in (0, 1).
           scope:            AgeScope restricting which persons contribute to the
                             measured prevalence, e.g. AgeScope(65, None) for age>=65.
           popType:          PopulationType.NHANES is supported. Kaiser inlines its
                             prevalence calls in person_factory.get_kaiser_person and
                             does not honor riskScaling; refused.
           peopleArgs:       dict forwarded to get_nhanes_people(**peopleArgs). An
                             outcomePrevalenceModelRepository key, if present, is dropped
                             (the calibrator manages seeding itself).
           baselineRiskScaling: optional dict[OutcomeType, float] applied to other
                                outcomes for the duration of the calibration. Lets
                                callers chain calibrations (e.g., pin epilepsy at a
                                previously calibrated value while solving CV-for-stroke).
                                The entry for scaleOutcomeType, if present, is overridden.

           Implementation: constructs people once with no priorToSim seeding, snapshots
           each in-scope person's RNG state, and per brentq iteration restores the RNG,
           clears _outcomes, and re-runs Person.seed_prevalent_outcomes with the trial
           scaling. The inner loop is deterministic in s irrespective of nhanesWeights.

           Returns: float scaling to feed into prevalenceRiskScaling={scaleOutcomeType: s}.'''

        if popType is not PopulationType.NHANES:
            raise NotImplementedError(
                f"calibrate_prevalence only supports PopulationType.NHANES; "
                f"got {popType}."
            )
        if scaleOutcomeType in {OutcomeType.MI, OutcomeType.COGNITION}:
            raise ValueError(
                f"{scaleOutcomeType} prevalence model ignores riskScaling; cannot scale."
            )
        defaultOpmr = OutcomePrevalenceModelRepository(useDefaults=False)
        if not defaultOpmr.has_prevalence_model(scaleOutcomeType):
            raise ValueError(
                f"{scaleOutcomeType} has no prevalence model registered; nothing to scale."
            )
        if not defaultOpmr.has_prevalence_model(targetOutcomeType):
            raise ValueError(
                f"{targetOutcomeType} has no prevalence model registered; nothing to measure."
            )

        order = list(OutcomeType)
        if order.index(targetOutcomeType) < order.index(scaleOutcomeType):
            raise ValueError(
                f"targetOutcomeType {targetOutcomeType} is seeded before scaleOutcomeType "
                f"{scaleOutcomeType} in OutcomeType order; cannot cascade."
            )

        if not (0. < target < 1.):
            raise ValueError(
                f"target must be in (0, 1) for an empirical prevalence; got {target}."
            )

        peopleArgs = {k: v for k, v in peopleArgs.items() if k != "outcomePrevalenceModelRepository"}

        unseededPeople = PopulationFactory.get_nhanes_people(
            **peopleArgs, outcomePrevalenceModelRepository=None
        )
        scopePeople = [p for p in unseededPeople if scope.contains(p._current_age)]
        if len(scopePeople) == 0:
            raise ValueError(
                f"scope {scope} matched zero constructed persons; cannot calibrate against "
                f"an empty subset (check peopleArgs personFilters and n)."
            )

        rngStates = [copy.deepcopy(p._rng.bit_generator.state) for p in scopePeople]
        baseScaling = dict(baselineRiskScaling or {})

        def empiricalGap(logS):
            scaling = {**baseScaling, scaleOutcomeType: math.exp(logS)}
            opmr = OutcomePrevalenceModelRepository(riskScaling=scaling, useDefaults=False)
            hits = 0
            for person, state in zip(scopePeople, rngStates):
                person._rng.bit_generator.state = copy.deepcopy(state)
                for ot in person._outcomes:
                    person._outcomes[ot] = []
                person.seed_prevalent_outcomes(opmr)
                if person.has_outcome_prior_to_simulation(targetOutcomeType):
                    hits += 1
            return hits / len(scopePeople) - target

        # Bracket in log-space (scaling = exp(logS)): keeps scaling > 0 and is well-conditioned
        # for log-linear prevalence models. ±15 maps to ~3e-7..3e6, wide enough to span the
        # achievable prevalence floor/ceiling so brentq's sign-change requirement is met.
        lo, hi = -15.0, 15.0
        gLo, gHi = empiricalGap(lo), empiricalGap(hi)
        # Both endpoints same-sign => target is outside the achievable range, not a bracketing
        # failure; surface a specific error instead of letting brentq raise.
        if gLo > 0 and gHi > 0:
            raise ValueError(
                f"target {target} for {targetOutcomeType} in scope {scope} is below the "
                f"minimum achievable prevalence ({gLo + target:.4f}) by scaling "
                f"{scaleOutcomeType}."
            )
        if gLo < 0 and gHi < 0:
            raise ValueError(
                f"target {target} for {targetOutcomeType} in scope {scope} is above the "
                f"maximum achievable prevalence ({gHi + target:.4f}) by scaling "
                f"{scaleOutcomeType}. This often means scaling {scaleOutcomeType} does not "
                f"cascade into {targetOutcomeType}."
            )
        if gLo == 0:
            scaling = math.exp(lo)
        elif gHi == 0:
            scaling = math.exp(hi)
        else:
            # Brent's method: bracketed root finder combining bisection (guaranteed convergence)
            # with secant / inverse-quadratic steps (superlinear when well-behaved). xtol is on
            # logS, so 1e-4 ≈ 0.01% resolution on scaling — well below epidemiological precision.
            logScaling = brentq(empiricalGap, lo, hi, xtol=1e-4)
            scaling = math.exp(logScaling)

        print(f"calibrate_prevalence: scale={scaleOutcomeType.value} "
              f"target_outcome={targetOutcomeType.value} scope={scope.label} "
              f"target={target:.4f} scaling={scaling:.4f} scopeN={len(scopePeople)}")
        return scaling

    @staticmethod
    def get_state_population(proportion=0.01, year=2030, personFilters=None):
        people = PopulationFactory.get_state_people(proportion=proportion, year=year, personFilters=personFilters)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)

    @staticmethod
    def get_partitioned_nhanes_people(year=None, n=None):
        """NOT IN USE at the moment: get_nhanes_people fits its distributions on
           get_partitioned_nhanes_people_crude instead, which groups on the four categorical variables that
           matter most rather than on all nine. All nine span about 10800 combinations for the roughly 5400
           adults of a single year, so about 96% of the groups this returns cannot be fit a covariance
           matrix that is not singular, whatever n is.

           Partitions a NHANES df in all possible combinations of categorical variables that actually exist in NHANES.
           Group is defined as a specific combination of the categorical variables.
           Returns a dictionary: keys are the groups, values are dataframes with the NHANES rows for that particular group.

           The people partitioned here are drawn with the NHANES weights, so the size of each group,
           and with it the number of draws the group contributes later, follows the weighted population
           rather than the raw NHANES sample. That is why get_nhanes_people refuses to weight the draws
           a second time. n is the size of that weighted draw and defaults to fitSampleSize.

           Note that the weighted draw is made with replacement, so an n past the number of distinct
           NHANES rows of that year adds copies rather than new people. It settles the group sizes, but
           it cannot make a covariance matrix any less singular: the rank of a group's covariance is
           bounded by how many distinct people it holds, not by how many times they appear."""
        n = PopulationFactory.fitSampleSize if n is None else n
        pop = PopulationFactory.get_nhanes_population(n=n, year=year, personFilters=None, nhanesWeights=True)
        pop.advance(1)
        df = pop.get_all_person_years_as_df()
        dfForGroups = dict()
        #this approach is running the risk of missing some categories present in the df, eg by the use of range for antiHypertensiveCount
        #gender, smoking, raceEthnicity, statin, education, alcoholPerWeek, anyPhysicalActivity, antiHypertensiveCount
        #for ge, sm, ra, st, ed, al, a, an in product(NHANESGender, SmokingStatus, NHANESRaceEthnicity, [True, False], 
        #                                               Education, AlcoholCategory, [True, False], range(7)):
        for mo, ge, sm, ra, st, ed, al, a, an in product(set(df[StaticRiskFactorsType.MODALITY.value].tolist()),
                                                     set(df[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                     set(df[StaticRiskFactorsType.SMOKING_STATUS.value].tolist()),
                                                     set(df[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist()),
                                                     set(df[DefaultTreatmentsType.STATIN.value].tolist()),
                                                     set(df[StaticRiskFactorsType.EDUCATION.value].tolist()),
                                                     set(df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value].tolist()),
                                                     set(df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].tolist()),
                                                     set(df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value].tolist())):
            dfForGroup = df.loc[(df[StaticRiskFactorsType.GENDER.value]==ge) & 
                                (df[StaticRiskFactorsType.SMOKING_STATUS.value]==sm) &
                                (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==ra) &
                                (df[DefaultTreatmentsType.STATIN.value]==st) &
                                (df[StaticRiskFactorsType.EDUCATION.value]==ed) &
                                (df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]==al) &
                                (df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value]==a) &
                                (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]==an), :].copy()
            if dfForGroup.shape[0]>0:
                #dfForGroups[ge.value, sm.value, ra.value, st, ed.value, al.value, a, an] = dfForGroup
                dfForGroups[mo, ge, sm, ra, st, ed, al, a, an] = dfForGroup
        return dfForGroups

    @staticmethod
    def is_singular(cov):
       """Checks if a covariance matrix is singular or not."""
       return True if not np.all(np.linalg.eig(cov)[0]>10**(-3)) else False

    @staticmethod
    def get_distributions(dfForGroups):
        """NOT IN USE at the moment: it goes with get_partitioned_nhanes_people, and get_nhanes_people now
           uses get_distributions_crude instead, which resolves a singular fit by pooling the group over all
           education levels rather than by borrowing another group's distribution.

           Fits a multivariate normal to the continuous variables of each specific combination of categorical variables (group).
           Returns a dictionary: keys are 'mean', 'cov', 'min', 'max', 'singular'.
           Values for 'singular' are boolean depending on whether the covariance matrix for that key is singular or not.
           Values for all other keys are np arrays.
           The min and max are useful because we need to impose bounds on the draws, Gaussians extend to infinity...."""
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value]
        meanForGroups = dict()
        covForGroups = dict()
        minForGroups = dict()
        maxForGroups = dict()
        sizeForGroups = dict()
        singularForGroups = dict()
        namesForGroups = dict()
        for key in dfForGroups.keys():
            meanForGroups[key], covForGroups[key] = multivariate_normal.fit(np.array(dfForGroups[key][nhanesContinuousVariables]))
            singularForGroups[key] = PopulationFactory.is_singular(covForGroups[key])
            minForGroups[key] = np.min(np.array(dfForGroups[key][nhanesContinuousVariables]), axis=0)
            maxForGroups[key] = np.max(np.array(dfForGroups[key][nhanesContinuousVariables]), axis=0)
            sizeForGroups[key] = dfForGroups[key].shape[0]
            namesForGroups[key] = dfForGroups[key]["name"].tolist()
        distributions = {"mean": meanForGroups, "cov": covForGroups, "min": minForGroups, "max": maxForGroups, "singular": singularForGroups,
                         "size": sizeForGroups, "names": namesForGroups}
        distributions = PopulationFactory.get_alt_groups(distributions)
        return distributions

    @staticmethod
    def get_kaiser_distributions():
        meanForGroups = dict()
        covForGroups = dict()
        minForGroups = dict()
        maxForGroups = dict()
        singularForGroups = dict()
        sizeForGroups = dict()
        namesForGroups = dict()
        
        #kaiser population size
        popSize = 315142

        fileDir = get_absolute_datafile_path("kaiser")
        csvFiles = ['/kaiserMin.csv', '/kaiserMax.csv', '/kaiserMean.csv', '/kaiserCovariance.csv', '/kaiserWeight.csv']        
        (minDf, maxDf, meanDf, covDf, weightDf) = list(map(lambda x: PopulationFactory.get_kaiserDf(x), [fileDir+y for y in csvFiles]))
        
        catVariables = PopulationFactory.kaiser_variable_types[VariableType.CATEGORICAL.value]
        conVariables = PopulationFactory.kaiser_variable_types[VariableType.CONTINUOUS.value]
        
        for index, key in minDf[catVariables].iterrows():
            key = tuple(key.tolist())
            meanForGroups[key] = meanDf.loc[
                                (meanDf["modality"]==key[0]) &
                                (meanDf["gender"]==key[1]) &
                                (meanDf["raceEthnicity"]==key[2]) &
                                (meanDf["smokingStatus"]==key[3]) &
                                (meanDf["afib"]==key[4]) &
                                (meanDf["pvd"]==key[5]) &
                                (meanDf["statin"]==key[6]) &
                                (meanDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()[0]
            covForGroups[key] = covDf.loc[
                                (covDf["modality"]==key[0]) & 
                                (covDf["gender"]==key[1]) &
                                (covDf["raceEthnicity"]==key[2]) &
                                (covDf["smokingStatus"]==key[3]) &
                                (covDf["afib"]==key[4]) &
                                (covDf["pvd"]==key[5]) &
                                (covDf["statin"]==key[6]) &
                                (covDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()
            singularForGroups[key] = PopulationFactory.is_singular(covForGroups[key])
            minForGroups[key] = minDf.loc[
                                (minDf["modality"]==key[0]) &
                                (minDf["gender"]==key[1]) &
                                (minDf["raceEthnicity"]==key[2]) &
                                (minDf["smokingStatus"]==key[3]) &
                                (minDf["afib"]==key[4]) &
                                (minDf["pvd"]==key[5]) &
                                (minDf["statin"]==key[6]) &
                                (minDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()[0]
            maxForGroups[key] = maxDf.loc[
                                (maxDf["modality"]==key[0]) &
                                (maxDf["gender"]==key[1]) &
                                (maxDf["raceEthnicity"]==key[2]) &
                                (maxDf["smokingStatus"]==key[3]) &
                                (maxDf["afib"]==key[4]) &
                                (maxDf["pvd"]==key[5]) &
                                (maxDf["statin"]==key[6]) &
                                (maxDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()[0]
            sizeForGroups[key] = int(
                                 popSize * 
                                 weightDf.loc[
                                (weightDf["modality"]==key[0]) &
                                (weightDf["gender"]==key[1]) &
                                (weightDf["raceEthnicity"]==key[2]) &
                                (weightDf["smokingStatus"]==key[3]) &
                                (weightDf["afib"]==key[4]) &
                                (weightDf["pvd"]==key[5]) &
                                (weightDf["statin"]==key[6]) &
                                (weightDf["anyPhysicalActivity"]==key[7]), "weight"].to_numpy()[0])
            namesForGroups[key] = [f"{index}kaiserPerson{i}" for i in range(sizeForGroups[key])]
        distributions = {"mean": meanForGroups, "cov": covForGroups, "min": minForGroups, "max": maxForGroups, 
                         "singular": singularForGroups, "size": sizeForGroups, "names": namesForGroups}
        distributions = PopulationFactory.get_alt_groups(distributions)
        return distributions

    @staticmethod
    def get_alt_groups(distributions):
        """For every singular covariance matrix in the distributions dict, finds an alternative distribution, a similar one,
        with a non-singular covariance matrix.
        The term 'similar' can be defined in many different ways..."""
        altForSingular = dict()
        for key in distributions["singular"].keys():
            if distributions["singular"][key]:
                altKeys = list()
                altProbs = list()
                meanOfSingular = distributions["mean"][key]
                for altKey in distributions["singular"].keys():
                    if not distributions["singular"][altKey]:
                       altProbability = multivariate_normal(distributions["mean"][altKey],
                                                            distributions["cov"][altKey], allow_singular=False).pdf(meanOfSingular)
                       altKeys += [altKey]
                       altProbs += [altProbability]
                #using the max probability means we are using both the mean and the sd of the alternative distribution
                altForSingular[key] = altKeys[altProbs.index(max(altProbs))]
        distributions["alt"] = altForSingular
        return distributions

    @staticmethod
    def draw_from_distributions(distributions, maxDraws=None):
        """Draws from the multivariate normal distributions for each combination of categorical variables (group).
        If a draw includes a continuous variable value outside the bounds, it re-draws.
        For each group, the number of draws from the distribution is equal to the number of people in that group in
        the original NHANES dataframe (as contained in dfForGroups).

        maxDraws: budget on the number of draws made for a single group, defaults to max(1000*size, 50000).
                  A group whose bounds its distribution never satisfies would otherwise loop forever, so
                  exhausting the budget raises a RuntimeError that reports the observed acceptance rate.
                  The budget is deliberately generous because low acceptance rates are normal here: a group
                  with a singular covariance matrix draws from an alternative group's distribution while
                  keeping its own bounds, and in NHANES 1999 that leaves at least one group accepting about
                  0.12% of its draws (its own ages run 52-93 while the distribution it draws from is centered
                  at 21). Such a group needs a few thousand draws to fill, and the draws it does keep are far
                  out in the tail of the distribution they came from."""
        drawsForGroups = dict()
        namesForGroups = dict()
        #just use the "mean" for the keys
        for key in distributions["mean"].keys():
            size = distributions["size"][key]
            namesForGroups[key] = distributions["names"][key]
            #use either the original distribution or the alternative if the cov matrix is singular
            distKey = key if not distributions["singular"][key] else distributions["alt"][key]
            distMean = distributions["mean"][distKey]
            distCov = distributions["cov"][distKey]
            dist = multivariate_normal(distMean, distCov, allow_singular=False)
            #this determines which bounds we use if the cov matrix is singular...the original ones or the ones from the alternative distribution
            if distributions["singular"][key] & (size>4):
                distMin = distributions["min"][key]
                distMax = distributions["max"][key]
            else:
                distMin = distributions["min"][distKey]
                distMax = distributions["max"][distKey]

            nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value]
            drawsNeeded = size
            draws = None
            #the batch is always exactly the shortfall, so it never exceeds size and does not need to be
            #capped against the budget -- note that rowsOutOfBounds below relies on draws holding size rows
            groupMaxDraws = max(1000*size, 50000) if maxDraws is None else maxDraws
            drawn = 0
            #the logic about when to reshape can be improved probably...
            while drawsNeeded>0:
                if drawn>=groupMaxDraws:
                    accepted = size - drawsNeeded
                    raise RuntimeError(f"""Cannot draw {size} in-bounds draws for group {key}: kept {accepted} of
                                           {drawn} draws (acceptance rate: {accepted/drawn:.4f}). The bounds of this
                                           group are almost never satisfied by the distribution being drawn from
                                           (singular covariance: {distributions["singular"][key]}, drawing from
                                           group: {distKey}). Increase maxDraws if the group is this narrow by design.""")
                drawn += drawsNeeded
                if draws is None:
                    draws = dist.rvs(size=drawsNeeded)
                else:
                    if len(draws.shape)==1:
                        draws = draws.reshape((1, len(nhanesContinuousVariables)))
                    if (drawsNeeded==1):
                        draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded).reshape((1,distMean.shape[0]))), axis=0 )
                    else:
                        draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded)), axis=0 )
                if size==1:
                    draws = draws.reshape((1, distMean.shape[0]))
 
                #find which draws contain one or more continuous variables that is outside of the bounds
                rowsOutOfBounds = np.array([False]*size)
                for i, bound in enumerate(distMin):
                    rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]<0.9*bound)
                for i, bound in enumerate(distMax):
                    rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]>1.1*bound)
                #how many more draws we need in the next iteration
                drawsNeeded = size - np.sum(~rowsOutOfBounds)
                #keep the draws that have all continuous variables within the bounds
                draws = draws[~rowsOutOfBounds,:] 
            drawsForGroups[key] = draws
        return drawsForGroups, namesForGroups

    @staticmethod
    def get_df_from_draws(drawsForGroups, namesForGroups, popType=PopulationType.NHANES.value):
        """Converts the draws from the distributions to a Pandas df."""
        catVariables = PopulationFactory.variable_types(VariableType.CATEGORICAL.value, popType=popType)
        conVariables = PopulationFactory.variable_types(VariableType.CONTINUOUS.value, popType=popType)
        df = pd.DataFrame(data=None, columns= ["name"]+catVariables+conVariables)
        for key in drawsForGroups.keys():
            #names = dfForGroups[key]["name"].tolist()
            names = namesForGroups[key]
            size = drawsForGroups[key].shape[0]
            dfCont = pd.DataFrame(drawsForGroups[key])
            dfCont.columns = conVariables
            dfCat = pd.concat([pd.DataFrame(key).T]*size, ignore_index=True)
            dfCat.columns = catVariables
            dfForGroup = pd.concat( [pd.Series(names), dfCat, dfCont], axis=1).rename(columns={0:"name"})
            df = pd.concat([df,dfForGroup]) if not df.empty else dfForGroup
        df[DynamicRiskFactorsType.AGE.value] = round(df[DynamicRiskFactorsType.AGE.value]).astype('int')
        return df

    @staticmethod
    def get_nhanes_age_standardized_population(n, year):
        #nhanesDf is needed just for the index
        #nhanesDf = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        nhanesDf = PopulationFactory.get_nhanesDf() 
        standardizedPop = StandardizedPopulation(year=year)
        weights = standardizedPop.populationWeightedStandard
        #it is ok weights are merged with the entire nhanesDf, because pandas sampling takes into account the index of the series
        weights = pd.merge(nhanesDf, weights, how="left", on=["age", "gender"]).popWeight
        pop = PopulationFactory.get_nhanes_population(n=n, year=year, personFilters=None, nhanesWeights=False, distributions=False, customWeights=weights)
        return pop

    @staticmethod
    def get_cloned_people(person, n):
        return pd.Series([person.__deepcopy__() for i in range(n)])

    @staticmethod
    def apply_person_filters_on_df(personFilters, df):
        """Keeps the rows of df that every df-level filter accepts.

           The df can be the NHANES rows or, when drawing from the distributions, the rows drawn from
           them. The second carries only the variables in PopulationFactory.variable_types plus name and
           WTINT2YR, so a filter written against a column that exists only in NHANES works on one and not
           on the other; that is reported rather than left as a KeyError out of pandas."""
        if personFilters is not None:
            for filterName, personFilterFunction in personFilters.filters["df"].items():
                if df.shape[0] == 0: #an earlier filter left nothing, and apply on an empty df has no rows to align to
                    return df
                try:
                    df = df.loc[df.apply(personFilterFunction, axis=1)]
                except KeyError as e:
                    raise RuntimeError(f"""The df filter '{filterName}' asked for the variable {e}, which
                                           this dataframe does not have. Its columns are:
                                           {sorted(df.columns.tolist())}.""")
        return df

    @staticmethod
    def apply_categorical_person_filters_on_groups(dfForGroups, personFilters,
                                                   popType=PopulationType.NHANES.value):
        """NOT IN USE at the moment: it was an optimization for the group-by-group fitting that
           get_partitioned_nhanes_people did. get_nhanes_people no longer fits per group of all nine
           categorical variables, and it applies the df-level filters to the redrawn dataframe instead,
           so there are no groups to drop ahead of a fit.

           Drops the groups that the df-level filters reject on their categorical variables alone.

           A group is one combination of the categorical variables, so a filter that depends only on
           those variables gives the same answer for every person in the group and can be decided once,
           for the whole group, before any distribution is fit. Dropping such a group here saves fitting
           a multivariate normal that nothing would ever draw from, and it keeps groups that no draw
           could satisfy away from the draw budget in draw_from_distributions.

           Which filters those are is not declared anywhere, so it is observed instead: the filter is
           handed a VariablesUsedRecorder wrapping a row of the group, which records every variable the
           filter reads. If everything it read is in the categorical list of variable_types then its
           answer holds for the whole group; if it read a continuous variable as well, the group cannot
           be decided here and is left to the person-by-person filtering that happens later.

           A filter that mixes the two is still decided whenever its categorical half alone settles the
           answer, because python stops evaluating an expression as soon as the result is known: for
           "gender is male and sbp>126" a female group never reaches the sbp test, so only gender is
           recorded and the group is dropped, which is right since no woman passes that filter at any
           sbp. For a male group the sbp test does run, sbp is recorded, and the group is left for later.

           Returns the subset of dfForGroups that survives. Raises if nothing does, because that leaves
           no distribution to draw from at all."""
        if personFilters is None:
            return dfForGroups
        catVariables = set(PopulationFactory.variable_types(VariableType.CATEGORICAL.value, popType=popType))
        groupsKept = dict()
        for key, dfForGroup in dfForGroups.items():
            #every person in the group shares the categorical values, so any row of it will do
            row = dfForGroup.iloc[0]
            keep = True
            for filterFunction in personFilters.filters["df"].values():
                recorder = VariablesUsedRecorder(row)
                try:
                    decision = filterFunction(recorder)
                except KeyError: #the filter wants a variable this df does not carry, leave it for later
                    continue
                #the answer holds for the whole group only if nothing outside the categorical list was read
                if (recorder.variablesUsed <= catVariables) and (not decision):
                    keep = False
                    break
            if keep:
                groupsKept[key] = dfForGroup
        if len(groupsKept) == 0:
            raise RuntimeError(f"""The df-level filters rejected all {len(dfForGroups)} groups on their
                                   categorical variables, so there is no distribution left to draw from.""")
        return groupsKept

    @staticmethod
    def apply_person_filters_on_people(personFilters, people):
        if personFilters is not None:
            for filterFunction in personFilters.filters["person"].values():
                people = pd.Series(list(filter(filterFunction, people)), dtype=object)
        return people

    @staticmethod
    def bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.NHANES.value, initializationModelRepository=None, outcomePrevalenceModelRepository=None, weights=None, maxDraws=None, distributions=None):
        """Samples rows from df until people holds exactly n Person-objects that pass personFilters.

           Filters drop a row only after it has been drawn -- the person-level ones only after a whole
           Person-object has been built from it -- so a draw of n rows usually yields fewer than n people
           and the shortfall has to be drawn again. Called with an empty people this is the whole draw,
           called with the people of an earlier draw it is the top-up of that draw.

           weights: the sampling weights (a Pandas Series aligned on the index of df, or None for uniform
                    sampling). Every pass here uses them, which is what keeps the people added by a later
                    pass drawn from the same distribution as the people of the first -- with weights=None
                    on a weighted population the top-up would silently bias the sample.
           distributions: when given, the continuous variables of every sampled row are redrawn from these
                    before the row becomes a Person, so that a row sampled more than once yields people
                    whose continuous variables differ. The df-level filters are then applied here too,
                    because the values they have to hold for are the drawn ones and those do not exist
                    until the row has been sampled.
           maxDraws: budget on the total number of rows sampled here, defaults to max(100*n, 500). Filters
                     that accept (almost) nothing would otherwise loop forever, so exhausting the budget
                     raises a RuntimeError that reports the observed acceptance rate."""
        if df.shape[0]==0:
            raise RuntimeError(f"""Cannot bring people to the target n={n}: the dataframe to sample from is empty.
                                   The df-level filters of personFilters rejected every row.""")
        maxDraws = max(100*n, 500) if maxDraws is None else maxDraws
        drawn = 0
        accepted = 0
        nRemaining = n - people.shape[0]
        while nRemaining>0:
            if drawn>=maxDraws:
                rate = accepted/drawn if drawn>0 else 0.
                raise RuntimeError(f"""Cannot bring people to the target n={n}: reached {people.shape[0]} people after
                                       sampling {drawn} rows (acceptance rate of the filters: {rate:.4f}). The filters
                                       of personFilters may be too restrictive, or incompatible with each other.
                                       Increase maxDraws if the filters are this restrictive by design.""")
            #the first pass has no information about how many draws survive the filters and so draws exactly
            #the shortfall, later passes size the draw with the acceptance rate observed so far (with a 20%
            #margin) so that restrictive filters converge in a few passes instead of one pass per person
            rate = accepted/drawn if accepted>0 else 1.
            margin = 1.2 if accepted>0 else 1.
            batch = min( int(np.ceil(nRemaining/rate*margin)), maxDraws-drawn )
            dfForPeople = df.sample(batch, replace=True, weights=weights)
            drawn += batch
            if distributions is not None:
                #each sampled row is redrawn on its own, so the people that came from one NHANES row differ
                dfForPeople = PopulationFactory.redraw_continuous_variables(dfForPeople, distributions)
                dfForPeople = PopulationFactory.apply_person_filters_on_df(personFilters, dfForPeople)
                if dfForPeople.shape[0]==0: #every drawn row was rejected, and apply on an empty df has no rows
                    continue
            peopleRemaining = pd.DataFrame.apply(dfForPeople, PersonFactory.get_person, popType=popType, initializationModelRepository=initializationModelRepository, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, axis="columns")
            peopleRemaining = PopulationFactory.apply_person_filters_on_people(personFilters, peopleRemaining)
            accepted += peopleRemaining.shape[0]
            people = pd.concat([people, peopleRemaining])
            nRemaining = n - people.shape[0]
        #the draws are iid so keeping the first n of an overshooting batch does not bias the sample
        return people.iloc[:n]

    @staticmethod
    def get_kaiser_people(n=1000, personFilters=None, wmhSpecific=None):
        '''The wmhSpecific variable is not needed in the function but it is passed on to the function from the trial.py
        because the NHANES get_nhanes_people function needs to get arguments from the trial.py.
        Creating Kaiser people is a time consuming process, in part due to the time needed to get the distributions.
        That is why we do that step only once and create a pandas dataframe only once.
        Since we need to plan for the possibility of using filters, and sometimes filters can be fairly restrictive,
        we need to use sampling with replacement from the dataframe. 
        It is unclear what memory needs we would have in order to create always a much larger sample than the one we need in
        simulations in order to avoid sampling with replacement.'''
        distributions = PopulationFactory.get_kaiser_distributions()
        drawsForGroups, namesForGroups = PopulationFactory.draw_from_distributions(distributions)
        df = PopulationFactory.get_df_from_draws(drawsForGroups, namesForGroups, popType=PopulationType.KAISER.value)
        df = PopulationFactory.apply_person_filters_on_df(personFilters, df)
        dfForPeople = df.sample(n, weights=None, replace=True)
        people = pd.DataFrame.apply(dfForPeople, PersonFactory.get_kaiser_person, axis="columns")
        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
        #weights=None because the initial Kaiser draw above is unweighted as well
        people = PopulationFactory.bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.KAISER.value, weights=None)
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_state_people(year=2030, personFilters=None, state="OH", samplingRate=0.025):
        '''Creates people as a representative part of a state's population a given year.
        The argument samplingRate indicates what proportion of the state's population we will simulate.
        Note that due to rounding by using sampling the increase in the size of the people creates is not proportional to the increase
        in the samplingRate.'''
        #df with only categorical variables completed
        dfWithCategoricals = PopulationFactory.get_dataframe_with_categoricals(year=year, state=state, samplingRate=samplingRate) 
        #get Gaussian distributions of continuous variables stratified...
        distributions = PopulationFactory.get_crude_distributions()
        #each row of dfWithCategoricals gets values for continuous variables based on the distributions
        df = PopulationFactory.append_dataframe_with_continuous(dfWithCategoricals, distributions)
        imr = InitializationModelRepository()
        opmr = OutcomePrevalenceModelRepository()
        people = pd.DataFrame.apply(df, PersonFactory.get_nhanes_person, args=(imr,), outcomePrevalenceModelRepository=opmr, axis="columns")
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_crude_distributions():
        '''Returns the Gaussians the continuous variables are drawn from, fit on the partition of NHANES
        by gender, race ethnicity, education and a 4-year age window.

        Built once and kept in _crudeDistributions. Neither the partition nor the fit takes an argument,
        so both give the same answer every time they are called, and together they cost about 5 seconds:
        partitioning scans the whole NHANES df once per combination of the four variables. Unlike
        get_nhanesDf this hands back the cached object itself rather than a copy, because everything
        downstream only reads from it.'''
        if PopulationFactory._crudeDistributions is None:
            partitionedNhanesDf = PopulationFactory.get_partitioned_nhanes_people_crude()
            PopulationFactory._crudeDistributions = PopulationFactory.get_distributions_crude(partitionedNhanesDf)
        return PopulationFactory._crudeDistributions

    @staticmethod
    def get_partitioned_nhanes_people_crude():
        '''Partitions the NHANES data, all rows, according to 4 categorical variables, the ones that are the most important overall for the prediction
        of continuous variables by using Gaussian distributions.
        Because the continuous variable distributions do not differ much for ages that are off by 1 or 2 years, use a range of ages and not just an exact age match.'''
        df = PopulationFactory.get_nhanesDf() 
        dictForCategoricals = dict()
        for gender, raceEthnicity, education, age in product(
                                                       set(df[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                       set(df[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist()),
                                                       set(df[StaticRiskFactorsType.EDUCATION.value].tolist()),
                                                       set(range(0,82,1))): #for age
            if age>1:
                ageMin = age-2
                ageMax = age+2
            elif age==1:
                ageMin = age-1
                ageMax = age+2
            elif age==0: #there is NOBODY in NHANES with age 0
                ageMin = age
                ageMax = age+2
            dfForCategoricals = df.loc[(df[StaticRiskFactorsType.GENDER.value]==gender) & 
                                       (df[DynamicRiskFactorsType.AGE.value].isin(list(range(ageMin,ageMax,1)))) & 
                                       (df[StaticRiskFactorsType.EDUCATION.value]==education) &
                                       (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==raceEthnicity), :]#.copy()
            if dfForCategoricals.shape[0]>0:
                dictForCategoricals[gender,raceEthnicity,education,age] = dfForCategoricals
        return dictForCategoricals

    @staticmethod
    def get_distributions_crude(dfForCategoricals):
        '''dfForCategoricals: a dictionary with keys gender, raceEthnicity, education, age and values a dataframe based on NHANES dataframe
        This function will attempt to fit Gaussian distributions for the continuous variables using this key and value 
        But if a singular Gaussian is created then the education level is removed from the key and the Gaussian is created by
        combining the values with all education levels
        If removing the education from the key does not provide a non-singular Gaussian we will need to fix it...'''
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        #during the simulation, age is treated as a continuous variable, but when we are given a population projection that includes ageGroup
        #ageGroup and age are treated as a categorical variable that we know
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)
        meanForCategoricals = dict()
        covForCategoricals = dict()
        singularForCategoricals = dict()
        minForCategoricals = dict()
        maxForCategoricals = dict()
        for key in dfForCategoricals.keys():
            meanForCategoricals[key], covForCategoricals[key] = multivariate_normal.fit(np.array(dfForCategoricals[key][nhanesContinuousVariables]))
            singularForCategoricals[key] = PopulationFactory.is_singular(covForCategoricals[key]) #some distributions might be singular
            minForCategoricals[key] = np.min(np.array(dfForCategoricals[key][nhanesContinuousVariables]), axis=0)
            maxForCategoricals[key] = np.max(np.array(dfForCategoricals[key][nhanesContinuousVariables]), axis=0)
        #keysToRemove = list() #these are the distributions that are singular
        keysSingular = list(filter(lambda x: singularForCategoricals[x], singularForCategoricals.keys()))
        for key in keysSingular:
            keyMinusEducation = tuple(list(key[0:2]) + [key[3]]) #key includes gender, race ethnicity, education, age
            if keyMinusEducation not in meanForCategoricals.keys(): #I might have done the fit on a prior pass
                allEducationKeys = [list(key[0:2]) + [ed.value, key[3]] for ed in Education]
                allEducationKeys = list(filter(lambda x: tuple(x) in list(dfForCategoricals.keys()), allEducationKeys))
                dfForAllEducationKeys = pd.concat([dfForCategoricals[tuple(edKey)] for edKey in allEducationKeys], ignore_index=True)
                meanForCategoricals[keyMinusEducation], covForCategoricals[keyMinusEducation] = multivariate_normal.fit(
                    np.array(dfForAllEducationKeys[nhanesContinuousVariables]))
                singularForCategoricals[keyMinusEducation] = PopulationFactory.is_singular(covForCategoricals[keyMinusEducation])
                minForCategoricals[keyMinusEducation] = np.min(np.array(dfForAllEducationKeys[nhanesContinuousVariables]), axis=0)
                maxForCategoricals[keyMinusEducation] = np.max(np.array(dfForAllEducationKeys[nhanesContinuousVariables]), axis=0)
                if singularForCategoricals[keyMinusEducation]: #if removing education does not create a non-singular Gaussian the process failed
                    raise RuntimeError("Process of creating non-singular Gaussian distributions has failed.")
            #keysToRemove.append(key)    
        #for key in keysToRemove:
        for key in keysSingular: 
            del singularForCategoricals[key]
            del meanForCategoricals[key]
            del covForCategoricals[key]
        distributions = {"mean": meanForCategoricals, "cov": covForCategoricals, "singular": singularForCategoricals,
                         "min": minForCategoricals, "max": maxForCategoricals}
        return distributions


    @staticmethod
    def get_dataframe_with_categoricals(year=2030, state="OH", samplingRate=0.01):
        '''Returns dataframe with complete categorical variables but no continuous variables, with each row
        corresponding to a single person.
        Because state population projections do not include information on default treatments, we will use NHANES data to partition each group to
        a meaningful default treatment group.'''
        df = PopulationFactory.get_stateDf(year=year, state=state)
        #partition the people to default treatments in a similar way as found in the nhanes data
        proportionForDefaultTreatments = PopulationFactory.get_proportionForDefaultTreatments()
        df['nForAgeAndDefaultTreatments'] = df.apply(lambda x: PopulationFactory.get_nForDefaultTreatments(
                                                                 x["ageGroup"], x["gender"], x["raceEthnicity"], x["statin"], 
                                                                 x["antiHypertensiveCount"], proportionForDefaultTreatments, x["nForAge"]), axis=1)
        df = df.loc[ (df["nForAgeAndDefaultTreatments"]>0) ] #keep only the rows that have 1 or more people
        df["name"] = np.arange(len(df)) #people with the same categorical variables will have the same name
        df["nForSampling"] = df["nForAgeAndDefaultTreatments"].apply(lambda x: range(math.floor(x*samplingRate + 0.5))) #this is how samplingRate influences the number of people created
        df = df.explode("nForSampling")
        df["modality"] = Modality.NO.value #all NHANES people will have the same modality
        return df 

    @staticmethod
    def get_stateDf(year=2030, state='OH'):
        '''Reads the CSV file that includes some categorical variables for each state and year and performs a bit of initial processing.
        Returns a dataframe that includes a portion of the microsim categorical variables and the number of people in that state by age.'''
        dataDir = get_absolute_datafile_path("state")
        data = pd.read_csv(dataDir+f"/pop_projection_{state.lower()}_{year}.csv")
        data[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value] = data[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)
        ageList5Years = [x for x in range(0,5)]
        ageList10Years = [x for x in range(0,10)] #this is for the last age group, the oldest
        data['age'] = data['ageGroup'].apply(lambda i: [x+(i-1)*len(ageList5Years) for x in ageList5Years] if i!=17 else 
                                                       [x+(i-1)*len(ageList5Years) for x in ageList10Years]  ) #not a typo
        data = data.explode('age')
        data["nForAge"] = data.apply(lambda x: PopulationFactory.get_nForAge_from_nForAgeGroup(x["age"],x["n"]), axis=1)
        data["statinAntihypertensiveCount"] = [[[st, an] for st in [True, False] for an in [0., 1., 2.]]] * len(data)
        data = data.explode("statinAntihypertensiveCount")
        data[["statin", "antiHypertensiveCount"]] = pd.DataFrame(data["statinAntihypertensiveCount"].tolist(), index=data.index)
        return data

    @staticmethod
    def get_nForAge_from_nForAgeGroup(age, nForAgeGroup):
        '''Returns the number of people we would reasonably expect to find with given age and number of people in age group.
        Uniform distribution over ages for ages less than 80 years old and a decreasing distribution after that.'''
        if age<80:
            return round(nForAgeGroup/5) #divides the number of people to equal parts for all ages in ageGroup
        elif age < 90:
            #the coefficients were obtained with arr = np.linspace(9, 0, 10) and normalized_arr = arr / arr.sum()
            proportionsForAgeDict = {80: 0.2, 81:0.17777778, 82: 0.15555556, 83: 0.13333333, 84: 0.11111111,
                                     85: 0.08888889, 86: 0.06666667, 87: 0.04444444, 88: 0.02222222, 89: 0.}
            return round(nForAgeGroup * proportionsForAgeDict[age]) #divides the number of people to decreasing parts as age increases
 
    @staticmethod
    def get_ageGroup_from_age(age):
        '''Returns age group given age.
        Age groups include 5 ages until age 79, and anyone older than 79 belongs to age group 17.'''
        if age>=80:
            return 17
        else:
            return age//5 + 1

    @staticmethod
    def get_nForDefaultTreatments(ageGroup, gender, raceEthnicity, statin, antiHypertensiveCount, proportionForDefaultTreatments, nForAge):
        '''Returns the number of people we expect to find with given statin  and antiHypertensiveCount from the number of 
        people with that age, the ageGroup, gender and raceEthnicity'''
        return int(round(proportionForDefaultTreatments[ageGroup, gender, raceEthnicity][statin, antiHypertensiveCount] * nForAge))

    @staticmethod
    def get_proportionForDefaultTreatments():
        '''For a given age group, gender, race ethnicity, statin, anti hypertensive count returns the proportion of NHANES people
        that have given statin and anti hypertensive count from all NHANES people with given age group, gender, race ethnicity.'''
        weightForTreatments = dict()
        proportionForTreatments = dict()
        df = PopulationFactory.get_nhanesDf() 
        df["ageGroup"] = df["age"].apply(lambda x: PopulationFactory.get_ageGroup_from_age(x))
        df["age"] = df["age"].astype(int)
        for ageGroup, gender, raceEthnicity in product(
                                                list(range(1,18,1)),
                                                #set(data[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                [ge.value for ge in NHANESGender],
                                                #set(data[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist())):
                                                [ra.value for ra in RaceEthnicity if ra.value!=6]): #NHANES does not include any asian...
            proportionForTreatments[ageGroup, gender, raceEthnicity] = dict()
            weightForTreatments = dict()
            sumForKey = 0
            for statin in [True, False]:
                for antiHypertensiveCount in [0., 1., 2.]:
                    dfForGroup = df.loc[
                                    (df["ageGroup"]==ageGroup) &
                                    (df[StaticRiskFactorsType.GENDER.value]==gender) & 
                                    (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==raceEthnicity) &
                                    (df[DefaultTreatmentsType.STATIN.value]==statin) &
                                    (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]==antiHypertensiveCount), :].copy()
                    if dfForGroup.shape[0]>0:
                        weightForTreatments[statin, antiHypertensiveCount] = sum(dfForGroup.loc[:,"WTINT2YR"].tolist())
                        sumForKey += weightForTreatments[statin, antiHypertensiveCount]
                    else:
                        weightForTreatments[statin, antiHypertensiveCount] = 0
            if sumForKey==0.:
                raise RuntimeError(f"Did not find NHANES people-data with gender {gender}, raceEthnicity {raceEthnicity}, age group {ageGroup}") 
            for statin in [True, False]:
                for antiHypertensiveCount in [0., 1., 2.]:
                    proportion =  weightForTreatments[statin, antiHypertensiveCount]/sumForKey if sumForKey>0. else 0.
                    proportionForTreatments[ageGroup, gender, raceEthnicity][statin, antiHypertensiveCount] = proportion
        return proportionForTreatments

    @staticmethod
    def redraw_continuous_variables(df, distributions):
        '''Replaces the continuous variables of every row of df with a draw from the distributions.

        The categorical variables and the age of each row are kept as they are, which is what makes this
        the same construction the state populations use: there the categorical variables and the age come
        from a state projection, here they come from the NHANES rows themselves, and in both cases only
        the continuous variables are drawn. The draw is made from a Gaussian fit on the few categorical
        variables that matter most for the continuous ones, and is then shifted to the mean of the group
        the row belongs to (see group_key_frame).'''
        df = df.copy()
        df["ageGroup"] = df[DynamicRiskFactorsType.AGE.value].apply(PopulationFactory.get_ageGroup_from_age)
        #age is not redrawn, it is one of the keys the distributions are stored under
        continuousToRedraw = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        continuousToRedraw.remove(DynamicRiskFactorsType.AGE.value)
        df = df.drop(columns=continuousToRedraw)
        return PopulationFactory.append_dataframe_with_continuous(df, distributions)

    @staticmethod
    def append_dataframe_with_continuous(dfWithCategoricals, distributions):
        '''Takes a dataframe where all categorical variables exist for each row, and uses the distributions to append columns
        with all continuous variables.
        The complete dataframe is returned.'''
        nhanesContinuousVariables = PopulationFactory.continuous_variables_drawn()
        #a draw is a point of its own distribution, and what is wanted is a point of the group the person
        #belongs to, so the draw is kept as its distance from the mean of the distribution it came from and
        #that distance is measured out from the mean of the group instead
        distKeysForRows = PopulationFactory.get_dist_keys_for_dataframe(dfWithCategoricals, distributions)
        drawMeans = np.empty((dfWithCategoricals.shape[0], len(nhanesContinuousVariables)))
        for distKey, rowsOfKey in distKeysForRows:
            drawMeans[rowsOfKey] = distributions["mean"][distKey]
        groupMeans = PopulationFactory.get_group_means_for_dataframe(dfWithCategoricals, drawMeans)
        #the shift is handed to the draw rather than added to the draw afterwards because the value that is
        #kept is the shifted one, so the shifted one is what the bounds have to hold for: a draw inside the
        #bounds of its distribution can be shifted outside of them, and outside of them meant a negative
        #trig, ldl, hdl or creatinine for 151 of the 5448 people of NHANES 1999 when the shift was applied
        #after the bounds had been checked
        draws = PopulationFactory.get_draws_for_dataframe(distKeysForRows, distributions, groupMeans - drawMeans)
        dfWithContinuous = pd.DataFrame(draws,
                                        columns=nhanesContinuousVariables,
                                        index=dfWithCategoricals.index)
        return pd.concat([dfWithCategoricals, dfWithContinuous], axis=1)

    @staticmethod
    def continuous_variables_drawn():
        '''The continuous variables that are drawn: all of them except age, which is not drawn but is one
        of the keys the distributions are stored under.'''
        continuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        continuousVariables.remove(DynamicRiskFactorsType.AGE.value)
        return continuousVariables

    @staticmethod
    def get_dist_keys_for_dataframe(dfWithCategoricals, distributions):
        '''Returns, for every distribution that has rows drawing from it, that distribution's key and the
        positions of those rows.

        The rows are handled one distribution at a time rather than one row at a time: every row with the
        same gender, race ethnicity, education and age draws from the same Gaussian, so that Gaussian is
        built once and asked for as many points as there are such rows.'''
        #the distributions do not go past age 80, older people draw from the age 80 distribution
        keyFrame = pd.DataFrame({
            StaticRiskFactorsType.GENDER.value: dfWithCategoricals[StaticRiskFactorsType.GENDER.value],
            StaticRiskFactorsType.RACE_ETHNICITY.value: dfWithCategoricals[StaticRiskFactorsType.RACE_ETHNICITY.value],
            StaticRiskFactorsType.EDUCATION.value: dfWithCategoricals[StaticRiskFactorsType.EDUCATION.value],
            DynamicRiskFactorsType.AGE.value: dfWithCategoricals[DynamicRiskFactorsType.AGE.value].clip(upper=80)})
        distKeysForRows = list()
        for key, rowsOfKey in keyFrame.groupby(list(keyFrame.columns), observed=True).indices.items():
            #a group whose own distribution was singular was fit without education, see get_distributions_crude
            distKey = key if key in distributions["mean"].keys() else (key[0], key[1], key[3])
            distKeysForRows.append((distKey, rowsOfKey))
        return distKeysForRows

    @staticmethod
    def get_draws_for_dataframe(distKeysForRows, distributions, shift):
        '''Draws the continuous variables of every row, one distribution at a time.

        distKeysForRows: the (distribution key, row positions) pairs of get_dist_keys_for_dataframe.
        shift: what is added to the draw of each row in order to move it from the mean of the distribution
               it was drawn from to the mean of the group the row belongs to. It is applied before the
               bounds are checked, see draw_within_bounds.

        Returns the drawn, shifted values.'''
        draws = np.empty_like(shift)
        clipped = 0
        for distKey, rowsOfKey in distKeysForRows:
            dist = multivariate_normal(distributions["mean"][distKey], distributions["cov"][distKey],
                                       allow_singular=False)
            draws[rowsOfKey], clippedForKey = PopulationFactory.draw_within_bounds(dist,
                                                                    distributions["min"][distKey],
                                                                    distributions["max"][distKey],
                                                                    len(rowsOfKey), shift=shift[rowsOfKey])
            clipped += clippedForKey
        if clipped > 0:
            print(f"""Warning: {clipped} of {draws.shape[0]} people had a group mean too far from the distribution
                      they were drawn from for the bounds of that distribution to be met, and were clipped to
                      those bounds. Their group holds too few people for its mean to be a reliable one.""")
        return draws

    @staticmethod
    def draw_within_bounds(dist, distMin, distMax, size, shift=None, maxAttempts=100):
        '''Returns size draws from dist, shifted by shift, redrawing the ones that fall outside the bounds.

        Gaussians extend to infinity while the people they were fit on do not, so a draw that no NHANES
        person comes close to is thrown away and made again. The bounds are widened by 10% first, so that
        the edges of the observed range stay reachable.

        shift, one row of it per draw, is added to a draw before its bounds are checked, because it is the
        shifted value that is kept and therefore the shifted value the bounds have to hold for. Rows of the
        same distribution do not share a shift, so each row is redrawn on its own until its own shifted
        draw is in bounds.

        maxAttempts bounds the number of redraws of a single row, because a shift can be large enough that
        almost no draw meets the bounds under it. A row that exhausts it keeps its last draw clipped into
        the bounds, and is counted in the second return value: such a shift comes from the mean of a group
        of very few people, and leaving that row at the edge of what NHANES holds is better than both
        raising on data that is otherwise fine and keeping a value that no person could have.

        Returns the draws and how many of them had to be clipped.'''
        lowerBound, upperBound = 0.9*distMin, 1.1*distMax
        shift = np.zeros((size, len(distMin))) if shift is None else shift
        draws = np.empty((size, len(distMin)))
        pending = np.arange(size)
        for attempt in range(maxAttempts):
            newDraws = np.atleast_2d(dist.rvs(size=pending.shape[0])) + shift[pending]
            inBounds = ((newDraws >= lowerBound) & (newDraws <= upperBound)).all(axis=1)
            draws[pending[inBounds]] = newDraws[inBounds]
            pending = pending[~inBounds]
            if pending.shape[0] == 0:
                return draws, 0
            #the rows that are still pending keep this draw if the attempts run out, hence the bookkeeping
            lastDraws = newDraws[~inBounds]
        draws[pending] = np.clip(lastDraws, lowerBound, upperBound)
        return draws, pending.shape[0]

    @staticmethod
    def get_group_means_for_dataframe(dfWithCategoricals, drawMeans):
        '''Returns, for every row of dfWithCategoricals, the mean of its group, and for the rows whose
        group NHANES does not hold at all the mean of the distribution the row was drawn from, which
        leaves that row where its draw put it.'''
        groupMeans = PopulationFactory.look_up_group_means(dfWithCategoricals)
        return np.where(np.isnan(groupMeans), drawMeans, groupMeans)

    @staticmethod
    def look_up_group_means(dfWithCategoricals):
        '''Looks up the mean of the group of every row, np.nan for the rows whose group is not there.'''
        keyFrame = PopulationFactory.group_key_frame(dfWithCategoricals)
        return PopulationFactory.get_group_means().reindex(pd.MultiIndex.from_frame(keyFrame)).to_numpy()

    @staticmethod
    def group_key_frame(df):
        '''The variables a group is defined by, read off any df that carries the categorical variables.

        Both the means and the look-up of those means are built from this one function, so the key that
        the means are stored under and the key they are asked for cannot drift apart. The columns are
        cast because the two sides do not always arrive with the same dtype -- the age of a state
        projection, for one, comes out of an explode as object -- and a MultiIndex whose level dtype
        differs matches nothing, which would silently leave every row unshifted.

        Which variables these are is a compromise. A group has to be fine enough to say something the
        crude distribution does not already say, and large enough for its mean to be worth using: the
        draw already conditions on gender, race ethnicity, education and age, so what a group adds is
        whether the person is on antihypertensives (which decides sbp and dbp) and whether they are
        physically active (bmi and waist). Both are taken as yes/no rather than as the count and any
        finer grouping, which keeps the median group at 54 people. Grouping on all nine categorical
        variables, as this used to, left the median group holding ONE person, so the "mean" was that
        person and the shift carried their noise into everyone drawn for that group.'''
        return pd.DataFrame({
            StaticRiskFactorsType.GENDER.value: df[StaticRiskFactorsType.GENDER.value].astype(int),
            StaticRiskFactorsType.RACE_ETHNICITY.value: df[StaticRiskFactorsType.RACE_ETHNICITY.value].astype(int),
            StaticRiskFactorsType.EDUCATION.value: df[StaticRiskFactorsType.EDUCATION.value].astype(int),
            "ageGroup": df[DynamicRiskFactorsType.AGE.value].apply(PopulationFactory.get_ageGroup_from_age).astype(int),
            "anyAntiHypertensive": (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value] > 0).astype(bool),
            DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)})

    @staticmethod
    def get_group_means():
        '''The mean of every continuous variable, for every group of NHANES people.

        This is all the draws need from those groups: a draw is re-centered on the mean of the group of
        the person it is for. A group is far too small to fit a covariance matrix on, which is why the
        distributions themselves are fit on a much coarser partition, but it is big enough for a mean.'''
        if PopulationFactory._groupMeans is None:
            df = PopulationFactory.get_nhanesDf()
            keyFrame = PopulationFactory.group_key_frame(df)
            PopulationFactory._groupMeans = df[PopulationFactory.continuous_variables_drawn()].groupby(
                                                [keyFrame[column] for column in keyFrame.columns],
                                                observed=True).mean()
        return PopulationFactory._groupMeans

    @staticmethod
    def get_draws_from_distributions_adjusted(row, distributions, nhanesDfPartitioned, nhanesDfPartitionedByYear=None):
        #NOT IN USE at the moment: append_dataframe_with_continuous does this for the whole dataframe at
        #once rather than one row at a time, see get_draws_for_dataframe and get_group_means_for_dataframe.
        #It needed a partition of NHANES into 19365 dataframes only to take a mean from a few of them, and
        #recomputed that mean for every person of a group.
        '''For each group of categorical variables, first obtains values for the continuous variables by using distributions only from the
        main 3 or 4 categoricals and then shifts the draw for the continuous variables to an amount equal to the difference of the means
        between the NHANES people that match all categorical variables and the distribution of the NHANES people using only the 3 or 4 categoricals.
        This is the method for adjusting the fact that we used a crude distribution to do the draw, using only the most important 3 or 4 categoricals.'''
        draws = PopulationFactory.get_draws_from_distributions_crude(row, distributions)

        ageGroup = row["ageGroup"]
        ge = row["gender"]
        sm = row["smokingStatus"]
        ra = row["raceEthnicity"]
        ed = row["education"]
        al = row["alcoholPerWeek"]
        a = row["anyPhysicalActivity"]
        st = row["statin"]
        an = row["antiHypertensiveCount"]
        age = row["age"]
        
        if age>80:
            age=80
            ageGroup=16

        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)

        distKey = (ge, ra, ed, age)
        distKey = distKey if distKey in distributions["mean"].keys() else (ge, ra, age)
        distMean = distributions["mean"][distKey]

        #the mean the draw is shifted to comes from the narrowest group that exists: the group of this
        #survey year first, so that a 1999 person is not handed the levels of all years pooled together,
        #then the group over all years, and if even that group is not in NHANES then no shift at all
        groupKey = (ge, sm, ra, st, ed, al, a, an, ageGroup)
        dfForGroup = None
        if nhanesDfPartitionedByYear is not None:
            yearKey = groupKey + (row["year"],)
            dfForGroup = nhanesDfPartitionedByYear.get(yearKey)
        if dfForGroup is None:
            dfForGroup = nhanesDfPartitioned.get(groupKey)
        if dfForGroup is not None:
            meanOfGroup = np.mean(np.array(dfForGroup[nhanesContinuousVariables]), axis=0)
            meandiff = meanOfGroup - distMean
        else:
            meandiff= np.zeros(len(draws[0])) #if the specific group is not found at all in the Nhanes dataframe then just use the more crude estimate
        drawsShifted = np.array(draws[0]) + meandiff
        return drawsShifted

    def get_partitioned_nhanes_df_with_age_group(matchYear=False):
        '''NOT IN USE at the moment: the only thing the draws ever needed from these groups was the mean
        of each one, and get_group_means computes all of those at once instead of holding 19365
        dataframes to take a few means from. It also groups on fewer variables, see group_key_frame.

        Uses all NHANES data, from all years, and then partitions the dataframe to a dictionary where keys are the set of categorical variables and values
        are dataframes with all NHANES rows that correspond to the specific values of the categorical variables.

        matchYear appends the survey year to the key. The groups are then smaller, too small to fit a
        covariance matrix on, but large enough to take a mean from, which is what they are used for:
        the draws come from distributions pooled over all years and are shifted to the mean of a group,
        so keying that mean by year is what keeps a 1999 population from being handed 2017 levels.'''
        df = PopulationFactory.get_nhanesDf()
        df["ageGroup"] = df["age"].apply(lambda x: PopulationFactory.get_ageGroup_from_age(x))
        #grouping the df is what the loop below used to do one combination at a time: the product of the
        #nine variables is about 180 thousand combinations, nearly all of them empty, and each one used to
        #cost a pass over the whole df. Grouping visits the occupied combinations only, which is the same
        #set of keys the loop kept, in the same order.
        groupVariables = [StaticRiskFactorsType.GENDER.value,
                          StaticRiskFactorsType.SMOKING_STATUS.value,
                          StaticRiskFactorsType.RACE_ETHNICITY.value,
                          DefaultTreatmentsType.STATIN.value,
                          StaticRiskFactorsType.EDUCATION.value,
                          DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
                          DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                          DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value,
                          "ageGroup"]
        if matchYear:
            groupVariables = groupVariables + ["year"]
        return {key: dfForGroup for key, dfForGroup in df.groupby(groupVariables, observed=True)}

    def get_draws_from_distributions_crude(row, distributions):
        #NOT IN USE at the moment: its only caller is get_draws_from_distributions_adjusted, which is
        #itself unused. draw_within_bounds draws a whole group at a time instead of one point at a time.
        '''Uses categorical variable information to access the distribution of the continuous variables for that particular set of categoricals
        and then makes a draw from that distribution.
        If the draw ends up corresponding to a point in space where no NHANES data exist, that is no person has that extreme value(s),
        then the draw happens again.
        This function is only using a few basic categorical variables, hence the label "crude".
        For most groups, gender, race ethnicity, education and age we can get a non-singular gaussian distribution but in the few cases
        where we have a singular distribution then we use just gender, race ethnicity and age.'''
        gender = row["gender"]
        raceEthnicity = row["raceEthnicity"]
        education = row["education"]
        age = row["age"]

        if age>80:
            age=80
    
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)

        distKey = (gender, raceEthnicity, education, age)
        distKey = distKey if distKey in distributions["mean"].keys() else (gender, raceEthnicity, age)
    
        distMean = distributions["mean"][distKey]
        distCov = distributions["cov"][distKey]
        dist = multivariate_normal(distMean, distCov, allow_singular=False)
        distMin = distributions["min"][distKey]
        distMax = distributions["max"][distKey]
    
        drawsNeeded = 1 #1 draw per row, since each row represents one person
        size=drawsNeeded
        draws = None
        drawsForGroups = dict()
        while drawsNeeded>0:
            if draws is None:
                draws = dist.rvs(size=drawsNeeded)
            else:
                if len(draws.shape)==1:
                    draws = draws.reshape((1, len(nhanesContinuousVariables)))
                if (drawsNeeded==1):
                    draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded).reshape((1,distMean.shape[0]))), axis=0 )
                else:
                    draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded)), axis=0 )
            if drawsNeeded==1:
                draws = draws.reshape((1, distMean.shape[0]))
            #find which draws contain one or more continuous variables that is outside of the bounds
            rowsOutOfBounds = np.array([False]*size)
            for i, bound in enumerate(distMin):
                rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]<0.9*bound)
            for i, bound in enumerate(distMax):
                rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]>1.1*bound)
            #how many more draws we need in the next iteration
            drawsNeeded = size - np.sum(~rowsOutOfBounds)
            #keep the draws that have all continuous variables within the bounds
            draws = draws[~rowsOutOfBounds,:] 
        return draws







            
















                       












