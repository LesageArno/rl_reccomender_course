from .fuzzifier import NeighbourResumeFuzzifier, SimpleFuzzifier, OnTaxonomyFuzzificationMethod, AssociationsMethod
from ....Taxonomy.taxonomy_explorer import LEVEL_COLS 
import pandas as pd
import json
from typing import Any, overload
import numpy as np
from copy import deepcopy
import time

class fuzzyEvaluator():
    def __init__(self, levels:dict[str,float], df_taxonomy:pd.DataFrame, lvlCols:list[str], skillIdCol:str="unique_id"):
        
        # Instantiate a floating fuzzifier and a fixed fuzzifier 
        self.fuzzifier:NeighbourResumeFuzzifier = NeighbourResumeFuzzifier(levels, df_taxonomy, lvlCols, skillIdCol)
        self.simpleFuzzifier:SimpleFuzzifier = SimpleFuzzifier(levels)

        # Default resumes
        self.resumes:dict[str,list[int,Any]] = None
        
        # A random number generator
        self.rng:np.random.RandomState = np.random.RandomState()

    def resumesToLists(self, resumesDict:dict[str,list[int,Any]]) -> list[list[str,int,Any]]:
        # Initialise a list of resumes
        resumesLists = []
        
        # For every resumes
        for key, skills in resumesDict.items():
            for skill, expertise in skills:
                # Register each skills as [id, skill, expertise]
                resumesLists.append([key, skill, expertise])
        
        # Return a list of skills in the form [[id1, s1, e1],[id2, s2, e2],...,[idn, sn, en]]
        return resumesLists
    
    def resumesToDict(self, resumesList:list[list[str,int,Any]]) -> dict[str,list[int,Any]]:
        # Initialise the dict that will contain the resumes 
        resumesDict = {}
        
        # For each skills appearing, register it using the key
        for id, skill, expertise in resumesList:
            resumesDict[id] = resumesDict.get(id, [])
            resumesDict[id].append([skill, expertise])
        return resumesDict
    
    @overload
    def sortResumes(self, resumes:dict[str,list[int,Any]]) -> dict[str,list[int,Any]]: ...
    @overload
    def sortResumes(self, resumes:list[list[str,int,Any]]) -> list[list[str,int,Any]]: ...
    def sortResumes(self, resumes):
        # If the instance is a dict
        if isinstance(resumes, dict):
            resumesDictSorted = {}
            # Sort the dictionary using the skill id for every key
            for key, skills in resumes.items():
                resumesDictSorted[key] = sorted(skills, key = lambda x: x[0])
            return resumesDictSorted
        
        # If it is a list
        elif isinstance(resumes, list):
            # Sort directly the list using the uid and the id of the skill after
            return sorted(resumes, key=lambda x: (x[0], x[1]))
        
    def loadResume(self, resumes:dict[str,Any]) -> None:
        self.resumes = self.sortResumes(resumes)
    
    def loadFuzzyBaseline(self) -> None:
        # Make a baseline by fuzzifying the resumes using fixed fuzzification 
        self.fuzzyBaseline:dict[str,list[int,Any]] = self.simpleFuzzifier.fuzzify(self.resumes, unknownDefault=None)
        
        # Transform the resulting fuzzified dictionary into a list 
        self.fuzzyBaselineAsLists:list[list[str,int,Any]] = self.sortResumes(self.resumesToLists(self.fuzzyBaseline))
        
        # Get the one that are known 
        self.fuzzyBaselineIsKnown:list[bool] = [False if skill[2] is None else True for skill in self.fuzzyBaselineAsLists]
        
        # Get the number of known skills and the number of skills
        self.fuzzyBaselineKnownCount:int = sum(self.fuzzyBaselineIsKnown)
        self.fuzzyBaselineCount:int = len(self.fuzzyBaselineAsLists)
        
    def maskFuzzyBaseline(self, p:float = 0.1, seed:int = None) -> None:
        # Change the seed of the rng
        self.rng.seed(seed)
        
        # Get the values from which we can choose a mask
        possiblesMasks = np.where(self.fuzzyBaselineIsKnown)[0]
        
        # Get a portion of known skill to mask
        self.masked:np.ndarray[int] = self.rng.choice(possiblesMasks, size=int(p*self.fuzzyBaselineKnownCount),replace=False)
        
        # Create the masked array (replace expertise by None)
        self.maskedFuzzyBaselineAsLists = deepcopy(self.fuzzyBaselineAsLists)
        for mask in self.masked:
            self.maskedFuzzyBaselineAsLists[mask][2] = None
        
        # Sort the resulting list
        self.maskedFuzzyBaselineAsLists = self.sortResumes(self.maskedFuzzyBaselineAsLists)
        # And transform it as a dict
        self.maskedFuzzyBaseline = self.sortResumes(self.resumesToDict(self.maskedFuzzyBaselineAsLists))
        
        # Integrity check on the number of masks produced
        if (a:=sum([self.fuzzyBaselineAsLists[i]!=self.maskedFuzzyBaselineAsLists[i] for i in range(len(self.fuzzyBaselineAsLists))])) != int(self.fuzzyBaselineKnownCount*p):
            raise Exception(f"That's awkward. We do not have the proportion {p=:.4f} we wanted. We have {a} masked instead of the expected {int(self.fuzzyBaselineKnownCount*p)}")
    
    def computeRMSE(self, y:list[list[str,int,Any]], ypred:list[list[str,int,Any]]) -> float:
        # Check if the document have the same length
        if len(y) != len(ypred):
            raise Exception("The lists of skills should be the same (only diverging on expertises).")
        
        # Compute the sum of squared error
        SSE = 0
        for i in range(len(y)):
            # Check if the order is the same
            if y[i][:-1] != ypred[i][:-1]:
                raise Exception("The lists of skills must possess the same ordering.")
            # Update the Sum of Square Error
            SSE += (y[i][2] - ypred[i][2])**2
        
        # Compute Mean Squared Error and Root Mean Squared Error
        MSE = SSE/len(y)
        RMSE = MSE**0.5
        
        # Return RMSE
        return RMSE
    
    def evaluateOnTaxonomy(self, 
                   modes:list[OnTaxonomyFuzzificationMethod], 
                   P:list[float], 
                   seeds:list[int],
                   outPath:str = "out/onTaxonomyEvaluation.csv",
                   **modeParameters):
        
        # Load the baseline (simple fuzzification)
        self.loadFuzzyBaseline()
        
        # To inform the user
        count = 1
        begin = time.time()
        
        # Open the file
        with open(outPath, "w") as file:
            
            # Print the header
            file.write(f"mode,seed,p,")
            for param in modeParameters.keys():
                file.write(f"{param},")
            file.write("RMSE\n")
            
            # For each mode
            for mode in modes:
                # Print the mode
                print(f"[{count}, {time.time()-begin:.6f}s] {mode=}")
                # For each seed (experimentation)
                for seed in seeds:
                    
                    # For each probabilities
                    for p in P:    
                        # Compute the fuzzy baseline mask (choose to mask some known values). This will be compared onward with our results
                        self.maskFuzzyBaseline(round(p,5), seed)
                        
                        # Take only the one that will changes (the one we masked)  
                        maskedFilteredBaseline = [self.fuzzyBaselineAsLists[i] for i in self.masked] # That's the y to find again 
                        
                        # If mode is weighted, then use gamma
                        if mode in ["weighted", "weightedLog2"]:
                            # For each gamma
                            for gamma in modeParameters["gamma"]:
                                # Inform the user on the advancements
                                if count % 1000 == 0:
                                    print(f"[{count}, {time.time()-begin:.6f}s] {mode=}, {seed=}, p={round(p,5)}, gamma={round(gamma,5)}")
                                
                                # Fuzzify with a weighted methods, then transform the result to list and sort it
                                fuzzified = self.fuzzifier.fuzzify(self.maskedFuzzyBaseline, mode=mode, gamma=round(gamma,5))
                                fuzzified = self.sortResumes(self.resumesToLists(self.sortResumes(fuzzified)))
                                
                                # Retrieve only the one that changed (the masks we want to recapture)
                                maskedFilteredFuzzified = [fuzzified[i] for i in self.masked]
                                
                                # Write into the CSV the result
                                file.write(f"{mode},{seed},{round(p,5)},")
                                for param in modeParameters.keys():
                                    if param != "gamma":
                                        file.write(",")
                                    else:
                                        file.write(f"{round(gamma,5)},")
                                
                                # Compute the RMSE, save the results and flush (for some reason, it is necessary here)
                                file.write(f"{self.computeRMSE(maskedFilteredBaseline, maskedFilteredFuzzified)}\n")
                                file.flush()
                                count+=1
                        
                        # If it is not weighted, then proceed normally
                        else:
                            # Inform the user
                            if count % 1000 == 0:
                                print(f"[{count}, {time.time()-begin:.6f}s] {mode=}, {seed=}, ={round(p,5)}")
                            
                            # Apply fuzzification and transform the results into a sorted list 
                            fuzzified = self.fuzzifier.fuzzify(self.maskedFuzzyBaseline, mode=mode)        
                            fuzzified = self.sortResumes(self.resumesToLists(fuzzified))
                            
                            # Retrieve the values at the position of the mask
                            maskedFilteredFuzzified = [fuzzified[i] for i in self.masked]
                            
                            # Store the results
                            file.write(f"{mode},{seed},{round(p,5)},")
                            for param in modeParameters.keys():
                                file.write(",")
                            file.write(f"{self.computeRMSE(maskedFilteredBaseline, maskedFilteredFuzzified)}\n")
                            file.flush()
                            count+=1

    def evaluateOnRulesAssociations(self, 
                   modes:list[AssociationsMethod], 
                   P:list[float], 
                   seeds:list[int],
                   thresholds:list[int],
                   outPath:str = "out/onRulesAssociationsEvaluation.csv",
                   lastUnknownFill:dict[OnTaxonomyFuzzificationMethod,None|dict[str,Any]] = {"weightedLog2":{"gamma":1}}):

        # Load the baseline
        self.loadFuzzyBaseline()
        
        # To inform the user
        count = 1
        begin = time.time()
        
        # Open the outfile        
        with open(outPath, "w") as file:
            # Put the header
            file.write("mode,method,seed,p,threshold,subparam,RMSE\n")
            
            # For each mode (crisp, min, weighted)
            for mode in modes:
                # Print the switch of modes
                print(f"[{count}, {time.time()-begin:.6f}s] {mode=}")
                
                # For every seeds
                for seed in seeds:
                    # For each mask
                    for p in P:
                        # Compute the fuzzy baseline mask (choose to mask some known values). This will be compared onward with our results
                        self.maskFuzzyBaseline(round(p,5), seed)
                        
                        # Take only the one that will changes (the one we masked)  
                        maskedFilteredBaseline = [self.fuzzyBaselineAsLists[i] for i in self.masked] # That's the y to find again 
                        
                        # Load the association matrix earlier to gain some time
                        self.fuzzifier.loadAssociationRuleMatrix(self.maskedFuzzyBaseline, association=mode)
                        
                        # For every threshold
                        for k in thresholds:
                            # Fuzzify using association rules (and without recomputing the association rule matrix)
                            partialFuzzified = self.fuzzifier.fuzzify(self.maskedFuzzyBaseline, "associationRules", frequencyThreshold=k, loadARMatrix=False)
                            
                            # For the remaining unknown, fill using the desired method
                            for method in lastUnknownFill.keys():
                                if count % 100 == 0:
                                    print(f"[{count}, {time.time()-begin:.6f}s] {mode=}, {seed=}, {method=}, {k=}, p={round(p,5)}")

                                # Fuzzify with or without parameters
                                if lastUnknownFill[method] is None:
                                    fuzzified = self.fuzzifier.fuzzify(partialFuzzified, method)
                                else:
                                    fuzzified = self.fuzzifier.fuzzify(partialFuzzified, method, **lastUnknownFill[method])
                                
                                # Sort
                                fuzzified = self.sortResumes(self.resumesToLists(fuzzified))
                                # Retrieve the values at the position of the mask
                                maskedFilteredFuzzified = [fuzzified[i] for i in self.masked]
                                
                                # Write the results
                                file.write(f"{mode},{method},{seed},{round(p,5)},{k},{lastUnknownFill[method]},{self.computeRMSE(maskedFilteredBaseline, maskedFilteredFuzzified)}\n")
                                file.flush()
                                count+=1    
        
            
if __name__ == "__main__":
    
    # Get the necessary files for evaluations
    with open("fuzzifiedData/fuzzy_mastery_levels.json") as file:
        fuzzyMasteryLevels = json.load(file)
    with open("data/resumes.json") as file: 
        content = json.load(file)
    
    taxonomy = pd.read_csv("data/taxonomy.csv")
    fuzzyEval = fuzzyEvaluator(fuzzyMasteryLevels, taxonomy, LEVEL_COLS, "unique_id")
    fuzzyEval.loadResume(content)
    
    
    modes = ["linear","log2","weighted","weightedLog2"]
    P = np.arange(0.01,1.01,0.01)
    N = 10
    gamma = np.arange(0,1.01,0.01)
    
    fuzzyEval.evaluateOnTaxonomy(modes=modes, P=P, seeds=list(range(N)), gamma=gamma)
    
    fuzzyEval.evaluateOnRulesAssociations(
        modes=["crisp","min","weighted"],
        P = np.arange(0.02,1.01,0.02),
        seeds=list(range(10)),
        thresholds=list(range(1,16)),
        lastUnknownFill = {
            "weightedLog2":{"gamma":1},
            "log2":None,
            "linear":None
        }
    )
    
