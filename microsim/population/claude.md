# Population Package Documentation

This document provides detailed guidance for working with the population layer in MICROSIM.

## Overview

The `population/` package implements the mid-level of the MICROSIM hierarchy:

```
Trial (experimental design)
  ↓ compares
Population (collection of Person objects)
  ↓ created by
PopulationFactory (NHANES / Kaiser / state builders)
  ↓ People managed via
Person (individual agent)
```

A `Population` instance is essentially a Pandas Series of `Person` objects plus a
self-consistent set of prediction models (`PopulationModelRepository`). It knows how to
advance its people through time in a default (usual-care) manner; a `Trial` injects
experimental treatment strategies on top of that default.

The **Person-first principle** applies throughout: meaningful per-person logic lives on
`Person`; `Population` maps or filters over `self._people` and delegates to `Person`
methods. `Population` methods should not re-implement logic that already exists on
`Person`.

## Directory Structure

- `__init__.py`: Package init. Re-exports `Population`, `PopulationFactory`,
  `PopulationModelRepository`, `PopulationRepositoryType`, `StandardizedPopulation`, and
  `InitializationRepository` so callers can do
  `from microsim.population import Population, PopulationFactory`. Internal modules use
  concrete-path imports (e.g. `from microsim.population.population import Population`) to
  avoid import cycles.
- `population.py`: The `Population` class. Stores `self._people` (Pandas Series of
  `Person` objects), `self._modelRepository` (dict keyed by `PopulationRepositoryType`
  values), and `self._waveCompleted`. Contains the simulation engine, outcome queries,
  incidence/prevalence calculations, age/sex standardization, treatment-strategy queries,
  person-year dataframe construction, and console reporting helpers.
- `population_factory.py`: `PopulationFactory` — static-method-only class that builds
  `Population` instances from NHANES data, Kaiser data, or state-level projections.
  Also contains `calibrate_prevalence`, a root-finding utility for matching priorToSim
  prevalence targets.
- `population_model_repository.py`: `PopulationRepositoryType` enum and
  `PopulationModelRepository` class. The enum defines the four repository keys;
  `PopulationModelRepository` wraps them in a single `_repository` dict.
- `initialization_repository.py`: `InitializationRepository` — supplies the initializers
  run once per person at construction time (GCP cognition model and QALY assignment).
- `standardized_population.py`: `StandardizedPopulation` — loads US mortality-table data
  and builds age-group distributions and per-capita weights used for age/sex-standardized
  rate calculations.

## Repository + Factory Pattern

`PopulationModelRepository` aggregates four sub-repositories, each accessed via the
`PopulationRepositoryType` enum:

| Enum member | `.value` string | Sub-repository type |
|---|---|---|
| `PopulationRepositoryType.STATIC_RISK_FACTORS` | `"staticRiskFactors"` | `CohortStaticRiskFactorModelRepository` |
| `PopulationRepositoryType.DYNAMIC_RISK_FACTORS` | `"dynamicRiskFactors"` | `CohortDynamicRiskFactorModelRepository` |
| `PopulationRepositoryType.DEFAULT_TREATMENTS` | `"defaultTreatments"` | `DefaultTreatmentModelRepository` |
| `PopulationRepositoryType.OUTCOMES` | `"outcomes"` | `OutcomeModelRepository` |

`Population.__init__` receives a `PopulationModelRepository` and stores its inner dict
as `self._modelRepository`. Every access to a sub-repository inside `Population` and
`PopulationFactory` uses `PopulationRepositoryType.<MEMBER>.value` as the key — never
a raw string.

`PopulationFactory` is a collection of `@staticmethod` methods. The two primary
population-creation methods are described in the next section. It also exposes lower-level
helpers (`get_nhanes_people`, `get_kaiser_people`) that return a Pandas Series of `Person`
objects without wrapping them in a `Population`.

## PopulationFactory Method Signatures

### NHANES population

```python
@staticmethod
def get_nhanes_population(
    n=None,
    year=None,
    personFilters=None,
    nhanesWeights=False,
    distributions=False,
    customWeights=None,
    riskScaling=None,
    prevalenceRiskScaling=None,
) -> Population:
```

Parameters:
- `n`: number of people to sample. Honored in every sampling mode — with `nhanesWeights`,
  with `customWeights`, and with neither (rows are then drawn uniformly). `n=None` means no
  sampling at all: every person of that year who passes the filters is returned. Both
  `nhanesWeights=True` and `customWeights` require an `n`.
- `year`: NHANES survey year; must be one of `{1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017}`,
  or `None` to use every survey year at once. Note that `nhanesWeights=True` with `year=None` weighs
  people from different years against each other, which is not what NHANES weights mean.
- `personFilters`: a `PersonFilter` instance; defaults to an adults-only (age >= 18) filter
  when `None`.
- `nhanesWeights`: if `True`, sample with NHANES survey weights (`WTINT2YR`); requires `n`.
- `distributions`: if `True`, keep the categorical variables and the age of each NHANES row and
  replace only its continuous variables with a draw. This is the construction the state populations
  use: the draw comes from a multivariate Gaussian fit on gender, race ethnicity, education and a
  4-year age window, pooled over all NHANES years, and is then shifted by the difference between the
  mean of the group matching *every* categorical variable and the mean of that crude group (see
  `get_draws_from_distributions_adjusted`). Grouping on four variables rather than all nine is what
  leaves enough people per group to fit a covariance that is not singular — all nine variables span
  ~10,800 cells for the ~5,400 adults of a single year, and ~96% of those fits come out singular.
  `personFilters` are honored: they are applied to the redrawn dataframe, so a filter such as
  SBP > 126 holds for the people that come back. `customWeights` is refused with `distributions=True`;
  `nhanesWeights` applies as it does without it, since the sampling of the NHANES rows still decides
  who the population is made of.
- `customWeights`: alternative Pandas Series of sampling weights; mutually exclusive with
  `nhanesWeights`; requires `n`.
- `riskScaling`: optional `dict[OutcomeType, float]` applied to per-outcome risk inside
  `OutcomeModelRepository`.
- `prevalenceRiskScaling`: optional `dict[OutcomeType, float]` applied to per-outcome
  priorToSim risk inside `OutcomePrevalenceModelRepository`.

### Kaiser population

```python
@staticmethod
def get_kaiser_population(
    n=1000,
    personFilters=None,
    wmhSpecific=True,
    riskScaling=None,
) -> Population:
```

Parameters:
- `n`: number of people to draw from Kaiser distributions (default 1000).
- `personFilters`: a `PersonFilter` instance; `None` means no filter.
- `wmhSpecific`: if `True`, uses WMH-specific CV outcome models in the
  `OutcomeModelRepository`.
- `riskScaling`: optional `dict[OutcomeType, float]` applied inside
  `OutcomeModelRepository`.

### Generic dispatcher

```python
@staticmethod
def get_population(popType: PopulationType, **kwargs) -> Population:
```

Routes to `get_nhanes_population`, `get_kaiser_population`, or `get_state_population`
based on `popType`. Passes `**kwargs` through to the chosen method unchanged.

### Prevalence calibration

```python
@staticmethod
def calibrate_prevalence(
    scaleOutcomeType,
    targetOutcomeType,
    target,
    scope,
    popType,
    peopleArgs,
    baselineRiskScaling=None,
) -> float:
```

Uses Brent's method in log-space to find the `OutcomePrevalenceModelRepository`
`riskScaling` on `scaleOutcomeType` such that the realized priorToSim prevalence of
`targetOutcomeType` (within `scope`) equals `target`. Returns the float scaling to pass
as `prevalenceRiskScaling={scaleOutcomeType: scaling}` in a subsequent call to
`get_nhanes_population`. Only `PopulationType.NHANES` is supported.

## Simulation Engine — Serial vs Parallel

```python
population.advance(years, treatmentStrategies=None, nWorkers=1)
```

- `nWorkers=1` calls `advance_serial`, which maps `Person.advance(...)` over
  `self._people` in a single thread.
- `nWorkers > 1` calls `advance_parallel`, which splits `self._people` into
  `nWorkers` sub-populations, farms them out to a `multiprocessing.Pool`, and
  re-concatenates the results.

Each `Person` has its own `_rng` (NumPy `default_rng` instance). Because sub-populations
are independent copies with independent RNG state, parallel execution is reproducible
provided the initial RNG seeds are fixed.

`_waveCompleted` tracks how many years the population has been advanced.
Wave numbering starts at -1 before the first advance; after advancing 1 year,
`_waveCompleted == 0`.

## Creating and Advancing a Population (Example)

```python
from microsim.population import Population, PopulationFactory
from microsim.outcomes.outcome import OutcomeType

# Build a 500-person NHANES 1999 population sampled with survey weights
pop = PopulationFactory.get_nhanes_population(
    n=500,
    year=1999,
    nhanesWeights=True,
)

# Advance 5 years (single-threaded)
pop.advance(5)

# Advance 5 years using 4 worker processes
pop.advance(5, nWorkers=4)

# Query outcomes
stroke_count = pop.get_outcome_count(OutcomeType.STROKE)
dementia_incidence = pop.get_outcome_incidence(OutcomeType.DEMENTIA)

# Print a baseline summary
pop.print_baseline_summary()
```

## Population Attributes and Properties

- `_people`: Pandas Series of `Person` objects. The full simulation state lives here.
- `_n`: population size at construction.
- `_waveCompleted`: integer; -1 before any advance, then increments by the `years`
  argument on each `advance` call.
- `_modelRepository`: dict keyed by `PopulationRepositoryType.*.value` strings; populated
  from the `PopulationModelRepository` passed to `__init__`.
- `_staticRiskFactors` (property): list of static risk factor names registered in the
  static-risk-factor sub-repository.
- `_dynamicRiskFactors` (property): list of dynamic risk factor names registered in the
  dynamic-risk-factor sub-repository.
- `_defaultTreatments` (property): list of default treatment names registered in the
  default-treatment sub-repository.

## InitializationRepository

`InitializationRepository.get_initializers()` returns a dict of two one-time
per-person initializers that run during `Person` construction:

- `"_gcp"`: baseline Global Cognitive Performance score via `GCPModel`.
- `"_qalys"`: initial QALY assignment via `QALYAssignmentStrategy`.

These are passed as `imr` to `PersonFactory.get_nhanes_person` and
`PersonFactory.get_kaiser_person`.

## StandardizedPopulation

`StandardizedPopulation(year=2016)` loads the US mortality table from
`data/us.1969_2017.19ages.adjusted.txt` and exposes:

- `ageGroups`: `{gender_value: [[age, ...], ...]}` — age-group membership lists by gender.
- `populationPercents`: `{gender_value: [proportion, ...]}` — share of the total standard
  population in each age group.
- `populationWeightedStandard`: Pandas DataFrame with `age`, `gender`, and `popWeight`
  columns; used for age-standardized NHANES sampling via
  `PopulationFactory.get_nhanes_age_standardized_population`.

`Population.calculate_mean_age_sex_standardized_incidence` uses `StandardizedPopulation`
internally; callers rarely need to instantiate it directly.

## Gotchas

1. **Repository access via enum, not raw strings.** Always use
   `PopulationRepositoryType.OUTCOMES.value` (which equals `"outcomes"`) as the dict
   key, never the bare string `"outcomes"`. This keeps accesses consistent and
   refactoring safe.

2. **Parallel execution copies the model repository.** `get_sub_populations` calls
   `get_pop_model_repository_copy()` for each worker. The copy constructor for
   `PopulationModelRepository` takes `(dynamicRiskFactorRepository, defaultTreatmentRepository,
   outcomeRepository, staticRiskFactorRepository)` in that order.

3. **Per-Person RNG for reproducibility.** Each `Person._rng` is independent; reproducible
   results in parallel mode require fixing the initial person-level RNG seeds before
   calling `advance_parallel`.

4. **`_waveCompleted` vs. Person wave numbering.** `Population._waveCompleted` counts total
   years advanced. Each `Person` also has its own `_waveCompleted`; the two can diverge if
   a person dies partway through (the Person's wave counter stops advancing, but the
   Population's does not).

5. **NHANES year validation.** `get_nhanes_population` raises `RuntimeError` for any year
   not in `{1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017}`. `year=None` is the one
   exception: it skips the year filter and uses every survey year at once.

6. **`nhanesWeights` and `customWeights` are mutually exclusive**, and `customWeights` is also
   refused with `distributions=True`. Both checks run before any other work, so an argument
   combination that cannot be honored fails immediately rather than after the fitting.

7. **`distributions=True` is slow.** It partitions every NHANES year on gender, race ethnicity,
   education and a 4-year age window, fits a Gaussian per group, and then redraws the continuous
   variables of every row one row at a time. The first call in a process costs roughly 20s and
   later ones about 8s, since `get_nhanesDf` is cached. Prefer `distributions=False` (default)
   for most simulations.

8. **Kaiser population attribute set differs from NHANES.** Kaiser includes `afib` and
   `pvd` as categorical variables that NHANES does not; Kaiser omits `education` and
   `alcoholPerWeek`. Code that iterates over population attributes via the
   `nhanes_pop_attributes` / `kaiser_pop_attributes` dicts must use the correct set for
   the population type.

9. **The top-up must be given the sampling weights.** Person-level filters run after the
   Person-objects are built, so `bring_people_to_target_n` redraws whatever they dropped.
   It samples with the `weights=` it is handed — pass the same weights used for the initial
   draw, otherwise the replacement people come from a different (unweighted) distribution
   than the rest of the population.

10. **Over-restrictive person filters raise instead of hanging.** `bring_people_to_target_n`
    stops after building `maxDraws` Person-objects (default `max(100*n, 500)`) and raises a
    `RuntimeError` reporting the observed acceptance rate. Filters that are this restrictive
    by design need an explicit larger `maxDraws`.

11. **`get_nhanesDf` is cached and hands out copies.** Building the frame — reading the `.dta`
    and converting the columns — takes about 14 seconds, and every population build needs it,
    so it is built once into `PopulationFactory._nhanesDf`. Each call returns `.copy()` of the
    cache, never the cached object, because `get_treatment_weights` and
    `get_partitioned_nhanes_people_crude` add an `ageGroup` column and recast `age` on what they
    get back. Never return the cached frame directly.

12. **The `name` column is the frame's own index.** `get_nhanesDf` renames the data file's
    `index` column to `name`, and that value is what `Person._name` holds. The rename previously
    targeted a `level_0` column that the data file does not have, so no `name` column existed at
    all and the `distributions=True` path died with a `KeyError`.

13. **`draw_from_distributions` bounds its redraws**, and only the Kaiser path uses it now. Each
    group gets `maxDraws` draws, default `max(1000*size, 50000)`, and exhausting the budget raises
    a `RuntimeError` reporting the group and its acceptance rate. The budget is generous because a
    group whose covariance matrix is singular draws from an alternative group's distribution while
    keeping its own bounds, which can leave the acceptance rate near zero. The NHANES path avoided
    that problem rather than tuning around it, by grouping on four variables instead of nine (see
    the `distributions` parameter above).

14. **Two partitions of NHANES exist, for two different jobs.**
    `get_partitioned_nhanes_people_crude` groups on gender, race ethnicity, education and a 4-year
    age window and is what the Gaussians are fit on; `get_partitioned_nhanes_df_with_age_group`
    groups on all nine categorical variables plus an age group and is used only to compute the mean
    the crude draw is shifted to. The second one groups the dataframe rather than looping over the
    ~180,000 combinations of nine variables, nearly all of which are empty.

## Integration with the Core Framework

Populations sit between the `Person` layer and the `Trial` layer:

```
Person.advance(years, dynamicRiskFactorRepo, defaultTreatmentRepo,
               outcomeModelRepo, treatmentStrategies)
  ↑ called by
Population.advance_serial / advance_parallel
  ↑ called by
Trial.run()  (once for the control arm, once for the treated arm)
```

`PopulationFactory` is also called directly from `TrialDescription` subclasses
(`NhanesTrialDescription`, `KaiserTrialDescription`) to build the people that populate
trial arms.
