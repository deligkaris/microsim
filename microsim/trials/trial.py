from microsim.population import PersonListPopulation
from microsim.trials.trial_utils import get_analysis_name
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import copy
import pandas as pd

class Trial:
    def __init__(self, trialDescription, targetPopulation):
        self.trialDescription = trialDescription
        self.trialPopulation = self.select_trial_population(targetPopulation, 
            trialDescription.inclusionFilter, trialDescription.exclusionFilter)
        # select our patients from the population
        self.maxSampleSize = pd.Series(trialDescription.sampleSizes).max()
        self.treatedPop, self.untreatedPop = self.randomize(trialDescription.randomizationSchema)
        self.analyticResults = {}

    def select_trial_population(self, targetPopulation, inclusionFilter, exclusionFilter):
        filteredPeople = list(filter(inclusionFilter, list(targetPopulation._people)))
        return PersonListPopulation(filteredPeople)

    def randomize(self, randomizationSchema):
        treatedList = []
        untreatedList = []
        randomizedCount = 0
        # might be able to make this more efficient by sampling from the filtered people...
        for i, person in self.trialPopulation._people.items():
            while randomizedCount < self.maxSampleSize: 
                if not person.is_dead():
                    if randomizationSchema(person):
                        treatedList.append(copy.deepcopy(person))
                    else:
                        untreatedList.append(copy.deepcopy(person))
                    randomizedCount+=1
                else:
                    continue 
        return PersonListPopulation(treatedList), PersonListPopulation(untreatedList)
    
    def run(self):
        self.treatedPop._bpTreatmentStrategy = self.trialDescription.treatment
        lastDuration = 0
        for duration in self.trialDescription.durations:
            self.treatedDF, self.treatedAlive = self.treatedPop.advance_vectorized(duration-lastDuration)
            self.untreatedDF, self.untreatedAlive = self.untreatedPop.advance_vectorized(duration-lastDuration)
            self.analyze(duration, self.maxSampleSize, self.treatedPop._people.tolist(), self.untreatedPop._people.tolist())
            self.analyzeSmallerTrials(duration)
            
            lastDuration = duration

    def analyzeSmallerTrials(self, duration):
        for sampleSize in self.trialDescription.sampleSizes:
            numTrialsForSample = self.maxSampleSize // sampleSize
            for i in range(0, numTrialsForSample):
                if sampleSize==self.maxSampleSize:
                    continue
                sampleTreated = self.treatedPop._people.sample(int(sampleSize/2))
                sampleUntreated = self.untreatedPop._people.sample(int(sampleSize/2))
                self.analyze(duration, sampleSize, sampleTreated.tolist(), sampleUntreated.tolist())


    def analyze(self, duration, sampleSize, treatedPopList, untreatedPopList):
        for analysis in self.trialDescription.analyses:
            reg, se, pvalue, absoluteEffectSize = None, None, None, None
            try:
                reg, se, pvalue, absoluteEffectSize = analysis.analyze(treatedPopList, untreatedPopList)
            except PerfectSeparationError: # how to track these is not obvious, now now we'll enter "Nones"
                pass
            self.analyticResults[get_analysis_name(analysis, duration, sampleSize)] = {'reg' : reg, 
                                                                                        'se' : se, 
                                                                                        'pvalue': pvalue,
                                                                                        'absEffectSize' : absoluteEffectSize,
                                                                                        'duration' : duration, 
                                                                                        'sampleSize' : sampleSize,
                                                                                        'outcome' :  analysis.outcomeAssessor.get_name(),
                                                                                        'analysis' : analysis.name}
        return self.analyticResults
        
        

    