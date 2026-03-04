class IncidenceRateAnalysis:
    """Analysis class for calculating outcome incidence rates per 1000 person-years.

    This analysis computes incidence rates for both treated and control trial arms,
    enabling comparison of event rates adjusted for time at risk.
    """

    def __init__(self):
        pass

    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        """Calculate incidence rates per 1000 person-years for treated and control arms.

        Args:
            trial: Trial instance with treatedPop and controlPop
            assessmentFunctionDict: Dictionary with two required keys:
                - "outcome": function(population) -> list of booleans (event indicators)
                - "time": function(population) -> list of integers (person-years at risk)
            assessmentAnalysis: string identifier for this analysis type

        Returns:
            tuple: (treated_rate, control_rate) - both as events per 1000 person-years
        """
        outcomeFunc = assessmentFunctionDict["outcome"]
        timeFunc = assessmentFunctionDict["time"]

        # Get event indicators (boolean list) for each arm
        treatedOutcomes = list(map(outcomeFunc, [trial.treatedPop]))[0]
        controlOutcomes = list(map(outcomeFunc, [trial.controlPop]))[0]

        # Get person-years at risk (integer list) for each arm
        treatedPersonYears = list(map(timeFunc, [trial.treatedPop]))[0]
        controlPersonYears = list(map(timeFunc, [trial.controlPop]))[0]

        # Calculate rates: (events / person-years) * 1000
        treatedEvents = sum(map(int, treatedOutcomes))
        controlEvents = sum(map(int, controlOutcomes))

        treatedTotalPY = sum(treatedPersonYears)
        controlTotalPY = sum(controlPersonYears)

        treatedRate = 1000.0 * treatedEvents / treatedTotalPY if treatedTotalPY > 0 else 0.0
        controlRate = 1000.0 * controlEvents / controlTotalPY if controlTotalPY > 0 else 0.0

        return (treatedRate, controlRate)
