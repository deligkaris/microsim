from enum import Enum

# Q: should treatments be defined using their disease (eg hypertension),
# their effect (eg antihypertension), their drug class (eg statin), or what risk factors they
# affect (eg bp)?
#   treatment class, can affect outcome or risk factors
#   Do we have enough treatments in mind for how to organize these?


class DefaultTreatmentsType(Enum):
    STATIN = "statin"
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"


class CategoricalDefaultTreatmentsType(Enum):
    STATIN = "statin"


class ContinuousDefaultTreatmentsType(Enum):
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"


# nobody younger is on a default treatment: NHANES under-20s carry imputed ones (18-19 rates match
# the children's); 20 is also a boundary of the state 5-year age groups
DEFAULT_TREATMENT_MIN_AGE = 20

UNTREATED = {
    DefaultTreatmentsType.STATIN.value: False,
    DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
}
