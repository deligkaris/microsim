# Trial Framework Documentation

This document provides detailed guidance for working with the trial simulation framework in MICROSIM.

## Overview

The trial framework enables simulation-based clinical trial comparisons, allowing researchers to test different treatment strategies against control populations. Trials use statistical analysis methods (Cox regression, logistic regression, relative risk) to assess outcomes and compare interventions.

## Directory Structure

The `trials/` directory contains the experimental design framework:
- `trial.py`: Main trial orchestration
- `trial_description.py`: Configuration for trial setup
- `trial_type.py`: Trial type enumeration
- `trial_outcome_assessor.py`: Analysis and results computation
- `trial_outcome_assessor_factory.py`: Factory for creating outcome assessors
- Regression analysis modules:
  - `cox_regression_analysis.py`: Cox proportional hazards analysis
  - `logistic_regression_analysis.py`: Logistic regression for binary outcomes
  - `linear_regression_analysis.py`: Linear regression for continuous outcomes
  - `relative_risk_analysis.py`: Relative risk calculations
  - `regression_analysis.py`: Base regression analysis class
- `trialset.py`: Management of multiple related trials

## Trial Components

### Trial Orchestration

**trial.py** - The main Trial class that:
- Manages multiple populations (treatment arms)
- Coordinates simulation execution across all arms
- Delegates outcome assessment to TrialOutcomeAssessor

**trial_description.py** - Configuration object that specifies:
- Trial design parameters
- Population characteristics
- Treatment strategy assignments
- Analysis methods to apply

**trial_type.py** - Enumeration of supported trial types

### Outcome Assessment

**TrialOutcomeAssessor** - Analyzes trial results using:
- **Cox regression**: Time-to-event analysis with hazard ratios
- **Logistic regression**: Binary outcome analysis with odds ratios
- **Linear regression**: Continuous outcome analysis
- **Relative risk analysis**: Direct risk comparisons between arms

The assessor compares populations and generates statistical summaries of treatment effects.

## Running a Trial Simulation

### Step-by-Step Process

1. **Create population(s)**: Use `PopulationFactory.get_nhanes_population()` or `get_kaiser_population()`
   ```python
   control_pop = PopulationFactory.get_nhanes_population(n=1000, year=1999)
   treatment_pop = PopulationFactory.get_nhanes_population(n=1000, year=1999)
   ```

2. **Define treatment strategies**: Configure via `TreatmentStrategiesType` or use predefined strategies
   ```python
   from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType
   treatment_strategy = [TreatmentStrategiesType.BP_TREATMENT]
   ```

3. **Set up trial**: Use `Trial` class with `TrialDescription`
   ```python
   from microsim.trials.trial import Trial
   from microsim.trials.trial_description import TrialDescription

   description = TrialDescription(...)
   trial = Trial(description)
   ```

4. **Run simulation**: Call `population.advance(years)` or `trial.run()`
   ```python
   trial.run()
   ```

5. **Analyze results**: Use `TrialOutcomeAssessor` or population reporting methods
   ```python
   from microsim.trials.trial_outcome_assessor import TrialOutcomeAssessor
   assessor = TrialOutcomeAssessor(trial)
   results = assessor.assess()
   ```

## Testing Trial Components

Trial-related tests are located in `test/test_*.py` and include:
- Trial orchestration tests
- Outcome assessment validation
- Regression analysis verification
- Population comparison tests

Tests use the standard unittest framework with fixtures from `test/fixture/`.

## Integration with Core Framework

Trials sit at the top of the MICROSIM hierarchy:

```
Trial (experimental design)
  ↓ compares
Multiple Populations with different TreatmentStrategies
  ↓ analyzed by
TrialOutcomeAssessor (Cox regression, logistic regression, relative risk)
```

Each trial arm is a complete Population with its own:
- PersonFactory initialization
- TreatmentStrategyRepository
- Outcome tracking and reporting

## Common Trial Patterns

### Comparing Treatment Strategies

Create multiple populations with different treatment strategies, run them in parallel, and compare outcomes:
- Control arm: Standard of care (no additional treatment)
- Treatment arm(s): Specific intervention strategies

### Analyzing Multiple Outcomes

TrialOutcomeAssessor can analyze multiple outcome types simultaneously:
- Cardiovascular outcomes (STROKE, MI, CARDIOVASCULAR_DEATH)
- Cognitive outcomes (DEMENTIA, COGNITIVE_IMPAIRMENT)
- Overall mortality (DEATH)

### Statistical Methods Selection

Choose analysis method based on outcome type:
- **Time-to-event data**: Cox regression (most common for clinical trials)
- **Binary outcomes at fixed timepoint**: Logistic regression
- **Continuous outcomes**: Linear regression
- **Simple risk comparisons**: Relative risk analysis
