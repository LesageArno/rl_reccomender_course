from typing import Any, Literal

import taxonomy_explorer as te
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from collections.abc import Callable

# Profiling
import time

UNKNOWN_DEFAULT = {
    "data/courses.json": 0.6666666666666667,
    "data/jobs.json": 1,
    "data/resumes.json": 0.3333333333333333,
}

FuzzificationMethod = Literal["linear","weighted","log2","weightedLog2","associationRules"]
OnTaxonomyFuzzificationMethod = Literal["linear","weighted","log2","weightedLog2"]
AssociationsMethod = Literal["min","weighted","second"]

class SimpleFuzzifier():
    def __init__(self, levels:dict[str,float]):
        """Class to fuzzify an arbitrary dictionary.

        Args:
            levels (dict[str,float]): The registered level for fuzzification. Others are treated as unknown.
        """
        self.levels = levels
    
    def fuzzify(self, toFuzzify:dict[str,Any], unknownDefault:float|None = 0.5) -> dict[str,Any]:
        """Fuzzify the skills levels recursively by the parameters in `fuzzy_mastery_levels.json`. For unknown levels, the fuzzification is done by putting a particular default value. 

        Args:
            toFuzzify (dict[str,Any]): The dictionary whose values needs to be fuzzified.
            unknownDefault (float | None, optional): The default level whenever an unknown level is identified. Defaults to 0.5.

        Returns:
            dict[str,Any]: _description_
        """
        # If there is a dict, traverse the dict recursively
        if isinstance(toFuzzify, dict):
            return {k:self.fuzzify(v, unknownDefault) for k,v in toFuzzify.items()}
        # Same for the list
        elif isinstance(toFuzzify, list):
            return [self.fuzzify(v, unknownDefault) for v in toFuzzify]
        # Transform the labelled values into fuzzy values.
        elif isinstance(toFuzzify, (str, type(None))):
            return self.levels.get(toFuzzify, unknownDefault)
        # Return when everything is fuzzyfied
        else:
            return toFuzzify     

class NeighbourResumeFuzzifier(SimpleFuzzifier):
    def __init__(self, levels:dict[str,float], df_taxonomy:pd.DataFrame, lvlCols:list[str], skillIdCol:str="unique_id"):
        """A fuzzifier working for the resumes only. Do not provide others. Here, unknown values are fuzzified according to their neighbours in the given taxonomy.

        Args:
            levels (dict[str,float]): The named levels (levels that are known for the system. Outside of it, labels are treated as unknown 
            df_taxonomy (pd.DataFrame): The taxonomy to use.
            lvlCols (list[str]): The columns representing the levels of the taxonomy.
            skillIdCol (str, optional): The column representing the default id of the taxonomy. Defaults to "unique_id".
        """
        # Initialise as a child class
        super().__init__(levels)
        
        # Register the given data
        self.df_taxonomy:pd.DataFrame = df_taxonomy.copy()
        self.lvlCols = lvlCols.copy()
        self.skillIdCol = skillIdCol
        
        # "Compile" the taxonomy
        self.uid2name_taxonomy, self.name2uid_taxonomy = te.getTaxonomyID(self.df_taxonomy, self.lvlCols)
        self.bottomUpTaxonomy, self.topDownTaxonomy = te.getTaxonomy(self.df_taxonomy, self.name2uid_taxonomy, self.lvlCols)

        # Compile the different fuzzification functions
        self.mode = {
            "linear":self.getLinearAlpha,
            "weighted":self.getWeightedLinearAlpha,
            "log2":self.getLog2Alpha,
            "weightedLog2":self.getWeightedLog2Alpha,
            "associationRules":self.getARDictAlpha
        }
                
        # Dictionary for optimised bi-gram fuzzification
        self.associationDict = None
        self.associationFreqDict = None
        
    def getSkillsSets(self, searchDict:dict[str,Any], id:str) -> tuple[set[tuple[int, float]],set[tuple[int,None]]]:
        """For a given person (id) in searchDict, register the set of skills where level is known and the one for which the information is not known.

        Args:
            searchDict (dict[str,Any]): The dictionary to look into. 
            id (str): A key of this dictionary

        Returns:
            tuple[set[int],set[tuple[int, None]]]: A tuple with first element the skills with unknown level and in second the skills with known levels
        """
        # Initialise the variables
        unknownSet = set()
        knownSet = set()
        
        # For each skill of the person
        for skill in searchDict[id]:
            # Get if it is unknown
            if skill[1] is None:
                unknownSet.add(tuple(skill))
            # Get if it is known
            else:
                knownSet.add(tuple(skill))
        
        # Return the sets
        return unknownSet, knownSet 
    
    def getLinearAlpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], minUnknownLevel:float=0) -> list[list[int, float]]:
        """Function to guess the fuzzy belonging for each unknown level skills. It is computed as #(K&N)/#N where K is the set of known skills and N the set of neighbouring skills within the taxonomy.
        A neighbourhood is defined as all the skills that have the same direct subroot.
        
        Args:
            unknownSet (set[int]): The set of skills with unknown levels.
            knownSet (set[int]): The set of skills with known levels.
            minUnknownLevel (float): The minimal level of expertise given for a mentioned skill. Defaults to 0.

        Returns:
            list[list[int, float]]: A list with skills and updated skills levels.
        """
        # Initialise the skills
        skills = []
        
        # Get the direct id of all the known skills
        knownIDSet = set([i[0] for i in knownSet])
        
        # For all the skills
        for skill in unknownSet.union(knownSet):
            # If the skill as a level which is unknown
            if skill in unknownSet:
                # Get the set of neighbouring skills
                neighbourSet = set(te.getNeighbours(self.bottomUpTaxonomy, self.topDownTaxonomy, skill[0]))
                # Compute the alpha (the belonging to the fuzzy set of expertise).
                alpha = len(neighbourSet.intersection(knownIDSet))/len(neighbourSet)
            
            # If the skill is known, keep it as is.
            elif skill in knownSet:
                alpha = skill[1]
            
            # Register the skills
            skills.append([skill[0], max(minUnknownLevel,round(alpha,6))])
        return sorted(skills, key=lambda x: x[0])
    
    def getWeightedLinearAlpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], minUnknownLevel:float=0, gamma:float=0.5) -> list[list[int, float]]:
        """Function to guess the fuzzy mastery level of unknown level skills. It is computed as [∑α(N(i)&K)/#(N(i)&K)]*γ + (1-γ)*[#(N(i)&K)/#N(i)], with α being the truth value of some known skill level,
        K being the known skill set, N the neighbourhood of the unknown skill i, and γ some weight between 0 and 1.

        Args:
            unknownSet (set[tuple[int,None]]): The set of skills with unknown levels.
            knownSet (set[tuple[int,float]]): The set of skills with known levels.
            minUnknownLevel (float, optional): The minimum inferred value for unknown levels. If a known level is below this threshold, it is automatically replaced by the threshold. Defaults to 0.
            gamma (float, optional): The weight given to the skills levels average within the known part of the skill family . Defaults to 0.5.

        Returns:
            list[list[int, float]]: A list with skills and updated skills levels.
        """
        # Initialise the skills
        skills = []
        
        # Get a map of the ID of skills to their known levels, get the ID set of the skills with known levels
        knownIDtoLevelMap = {i[0]:i[1] for i in knownSet}
        knownIDSet = set([i for i in knownIDtoLevelMap.keys()])

        # For each skills
        for skill in unknownSet.union(knownSet):
            # If it is known, just register it
            if skill in knownSet:
                skills.append([skill[0], max(minUnknownLevel,skill[1])])
                continue
            
            # Otherwise, get the neighbours
            neighbourSet = set(te.getNeighbours(self.bottomUpTaxonomy, self.topDownTaxonomy, skill[0]))
            
            # Get the set for which known skills are in the neighbour set
            knownNeighbourSet = knownIDSet.intersection(neighbourSet)
            
            # Compute the alpha and append the skill.
            alpha = gamma * sum(knownIDtoLevelMap.get(i,0) for i in knownNeighbourSet)/max(1,len(knownNeighbourSet)) + (1-gamma) * len(knownNeighbourSet)/len(neighbourSet)
            skills.append([skill[0], max(minUnknownLevel, round(alpha, 6))])
        
        # Return the skills
        return sorted(skills, key=lambda x: x[0])
    
    def getLog2Alpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], minUnknownLevel:float=0) -> list[list[int, float]]:
        """Function to guess the fuzzy mastery level of unknown level skills. It is computed as [#(N(i)&K)*0.5]/log2(#N(i)), with α being the truth value of some known skill level,
        K being the known skill set, and N the neighbourhood of the unknown skill i.

        Args:
            unknownSet (set[tuple[int,None]]): The set of skills with unknown levels.
            knownSet (set[tuple[int,float]]): The set of skills with known levels.
            minUnknownLevel (float, optional): The minimum inferred value for unknown levels. If a known level is below this threshold, it is automatically replaced by the threshold. Defaults to 0.

        Returns:
            list[list[int, float]]: A list with skills and updated skills levels.
        """
        # Initialise the skills
        skills = []
        
        # Get the direct id of all the known skills
        knownIDSet = set([i[0] for i in knownSet])
        
        # For all the skills
        for skill in unknownSet.union(knownSet):
            # If the skill as a level which is unknown
            if skill in unknownSet:
                # Get the set of neighbouring skills
                neighbourSet = set(te.getNeighbours(self.bottomUpTaxonomy, self.topDownTaxonomy, skill[0]))
                # Compute the alpha (the belonging to the fuzzy set of expertise).
                # kts/nts
                kts = np.log2(1+len(neighbourSet.intersection(knownIDSet)))
                nts = max(1,np.log2(len(neighbourSet)))
                alpha = kts/nts
            
            # If the skill is known, keep it as is.
            elif skill in knownSet:
                alpha = skill[1]
            
            # Register the skills
            skills.append([skill[0], max(minUnknownLevel, round(alpha,6))])
        return sorted(skills, key=lambda x: x[0])
     
    def getWeightedLog2Alpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], minUnknownLevel:float=0, gamma:float=0.5) -> list[list[int, float]]:
        """Function to guess the fuzzy mastery level of unknown level skills. It is computed as [∑α(N(i)&K)/#(N(i)&K)]*γ + (1-γ)*[[#(N(i)&K)*0.5]/log2(#N(i))], with α being the truth value of some known skill level,
        K being the known skill set, N the neighbourhood of the unknown skill i, and γ some weight between 0 and 1.

        Args:
            unknownSet (set[tuple[int,None]]): The set of skills with unknown levels.
            knownSet (set[tuple[int,float]]): The set of skills with known levels.
            minUnknownLevel (float, optional): The minimum inferred value for unknown levels. If a known level is below this threshold, it is automatically replaced by the threshold. Defaults to 0.
            gamma (float, optional): The weight given to the skills levels average within the known part of the skill family . Defaults to 0.5.

        Returns:
            list[list[int, float]]: A list with skills and updated skills levels.
        """
        # Initialise the skills
        skills = []
        
        # Get a map of the ID of skills to their known levels, get the ID set of the skills with known levels
        knownIDtoLevelMap = {i[0]:i[1] for i in knownSet}
        knownIDSet = set([i for i in knownIDtoLevelMap.keys()])

        # For each skills
        for skill in unknownSet.union(knownSet):
            # If it is known, just register it
            if skill in knownSet:
                skills.append([skill[0],max(minUnknownLevel,skill[1])])
                continue
            
            # Otherwise, get the neighbours
            neighbourSet = set(te.getNeighbours(self.bottomUpTaxonomy, self.topDownTaxonomy, skill[0]))
            
            # Get the set for which known skills are in the neighbour set
            knownNeighbourSet = knownIDSet.intersection(neighbourSet)
            
            # Compute the alpha and append the skill.
            kts = np.log2(1+len(neighbourSet.intersection(knownIDSet)))
            nts = max(1,np.log2(len(neighbourSet)))             
            alpha = gamma * sum(knownIDtoLevelMap.get(i,0) for i in knownNeighbourSet)/max(1,len(knownNeighbourSet)) + (1-gamma) * (kts/nts)
            skills.append([skill[0], max(minUnknownLevel,round(alpha, 6))])
        
        # Return the skills
        return sorted(skills, key=lambda x: x[0])
    
    def getARDictAlpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], frequencyThreshold:int = 10):
        """Function to guess the real expertise level of unknown levels. It is computed based on the association rules retrieved by the `self.loadAssociationRuleDict`.
        In opposition to the other methods, this method might leave unknown if no associations rules were inferred for the unknown values. 

        Args:
            unknownSet (set[tuple[int,None]]): The set of skills with unknown levels.
            knownSet (set[tuple[int,float]]): The set of skills with known levels.
            frequencyThreshold (int, optional): The frequency of association observation from which we start to believe the inferred association rule . Defaults to 10.

        Returns:
            list[list[int, float]]: A list of skills with updated levels
        """
        # Get the skills in the known set and retrieve the id of known skills
        skills = [list(i) for i in knownSet]
        skillKnownID = {k[0] for k in knownSet}
        
        # Infer unknown if any
        if unknownSet != set():
            # For each unknown, match the highest rule in term of expertise (if the threshold is passed)
            for unkID, _ in unknownSet:
                possibilities = {self.associationDict[knownID][unkID] for knownID in skillKnownID if self.associationFreqDict[knownID][unkID] >= frequencyThreshold}
                possibilities.difference_update({None})
                skills.append([unkID, None] if len(possibilities) == 0 else [unkID, max(possibilities)])
        
        # Return the sorted version
        return sorted(skills, key = lambda x: x[0])

    def loadAssociationRuleDict(self, documents:dict[str,Any], association:AssociationsMethod = "weighted") -> None:
        """Generate an Association Rule matrix from the given document (in sparse format). In the end, you obtain a matrix displaying the number of occurrence of a given rule as well as another giving the expertise level we can expect from the association rule A[s1,s2] : s1 -> s2. 

        Args:
            documents (dict[str,Any]): The documents on which we will infer the association rule matrix.
            association (AssociationsMethod): The association method can be `crisp` in this case, whenever we see (s_i,e_i) -> (s_j, e_j) we register at A[i,j]: e_j. It can also be `weighted`: (s_i,e_i)->(s_j,e_j) => A[i,j] = e_i*e_j or `min`: (s_i,e_i)->(s_j,e_j) => A[i,j] = min(e_i,e_j) which corresponds to a fuzzy and. Defaults to "weighted"
        """
        # Create a sparse representation of the association rule matrix
        associationDict = defaultdict(lambda:defaultdict(lambda:None))
        associationFreqDict = defaultdict(lambda:defaultdict(lambda:1))
        
        # Fuzzify the documents
        documents = SimpleFuzzifier(self.levels).fuzzify(documents, None)
        
        # For every document, retrieve their skill-expertise set details
        for doc in documents.keys():
            unknownSet, knownSet = self.getSkillsSets(documents, doc)
            skillSet = unknownSet | knownSet
            
            # For every skill pairs in the skill-expertise set, if one expertise is unknown or skills are the same skip 
            for s1, e1 in skillSet:
                if e1 is None:
                    continue
                for s2, e2 in skillSet:
                    if s1==s2: 
                        continue
                    if e2 is None: 
                        continue
                    
                    # Update the frequencies
                    associationFreqDict[s1][s2] += 0 if associationDict[s1][s2] is None else 1
                    
                    # Initialise association dict when a rule is detected
                    associationDict[s1][s2] = 0 if associationDict[s1][s2] is None else associationDict[s1][s2]
                    
                    # Apply the desired association
                    if association == "weighted": 
                        associationDict[s1][s2] += e1*e2
                    elif association == "second": 
                        associationDict[s1][s2] += e2
                    elif association == "min": 
                        associationDict[s1][s2] += min(e1, e2)
        
        # Make the average over all keys in the dictionary 
        for k1, v1 in associationDict.items():
            for k2 in v1.keys():
                if associationDict[k1][k2] is not None:
                    associationDict[k1][k2] = round(associationDict[k1][k2]/associationFreqDict[k1][k2], 6)
                
                
        # Update the dictionary in self
        self.associationDict = associationDict
        self.associationFreqDict = associationFreqDict

    def fuzzify(self, toFuzzify:dict[str,Any], mode:FuzzificationMethod = "linear", **kwargs) -> dict[str,Any]:
        """Function to fuzzify the levels of Resume. Known levels are switched directly to their fuzzified version. Unknown level are estimated using 
        different function available with the parameter `mode`.

        Args:
            toFuzzify (dict[str,Any]): The dictionary containing the resume to fuzzify.
            mode (Literal[&quot;linear&quot;,&quot;weighted&quot;,&quot;log2&quot;,&quot;weightedLog2&quot;,&quot;associationRules&quot;], optional): A method to fuzzify the data. `linear` provide an estimate based on the number of known skills within the family of the unknown skill, `weighted` balance the preceding metrics with the average skill levels of the known skills within the family of the unknown skill, `log2` provide a version of 'linear', but such that the more you add skill within the family the less important it become. `weightedLog2` extends the last principle with the 'weighted' mode. Finally, `associationRules` base its analysis on the associations observed between two skills in the dataset. Defaults to "linear".
            kwargs: Some additional parameter for the different methods.
        Returns:
            dict[str,Any]: A fuzzified dictionary.
        """
        # Pass into the classical fuzzifier (to get rid of the known values)
        toFuzzify = SimpleFuzzifier(self.levels).fuzzify(toFuzzify, None)
        
        kwargs["loadARMatrix"] = kwargs.get("loadARMatrix", True)
        if mode == "associationRules" and kwargs["loadARMatrix"]:
            self.loadAssociationRuleDict(toFuzzify, association = kwargs.pop("association", "weighted"))
        kwargs.pop("loadARMatrix", None)
        kwargs.pop("association", None)
        
        # For each individuals
        for person in toFuzzify.keys():
            # Get the skills
            unknownSet, knownSet = self.getSkillsSets(toFuzzify, person)
            
            # Fuzzify the skills levels
            toFuzzify[person] = self.mode[mode](unknownSet, knownSet, **kwargs)
        
        return toFuzzify

    def sample_want_avoid(self, learners:dict[str,Any], minWant:int = 1, maxWant:int = 5, avoidDistribution:dict[int,float] = {0:0.6, 1:0.25, 2:0.15}, seed:int = 42, useLevel3:bool = False) -> dict[str, Any]:
        """Function to generate the preferences associated to each learner in the dataset.

        Args:
            learners (dict[str,Any]): The full learner dataset
            minWant (int, optional): The minimum number of skills a learner must want. Defaults to 1.
            maxWant (int, optional): The maximum number of skills (excluded) a learner might want. Defaults to 15.
            avoidDistribution (dict[int,float], optional): A distribution of the number of avoided skills.. Defaults to {0:0.6, 1:0.25, 2:0.15}.
            seed (int, optional): The seed reflecting the random choice of preference. Defaults to 42.
            useLevel3 (bool, optional): Whether we use the third level of the taxonomy instead of the fourth to select our preferences. Default to False.

        Returns:
            dict[str, Any]: A dict containing a list of pairs [skill,want] for all learner. -1 means skill should be avoided, 0 mean no constraint, 1 that the skill is wanted.
        """
        # Initialise a random number generator
        rng = np.random.RandomState(seed)
        
        # If level 3 use only level 3
        if useLevel3:
            # Retrieve the set of skills in Level 3
            S = set(np.arange(len(self.df_taxonomy["Type Level 3"].unique())).tolist())
        else:
            # List of skills in the taxonomy
            S = set(self.uid2name_taxonomy.keys())
        
        # Dictionary of results
        resultDict = {}
        
        # For every learner
        for l in learners.keys():
            # Set no preferences for every skills
            l_skillPreferences = {s:0 for s in S}
            
            # Select the number of skills we want and don't want
            n_want = rng.randint(minWant, maxWant)
            n_avoid = rng.choice(list(avoidDistribution.keys()), p=list(avoidDistribution.values()))
            
            # Get the index of the skills we want, remove them from the set of potential avoid
            l_index_want = set(rng.choice(list(S), n_want, replace=False))
            Sa = S.difference(l_index_want)
            
            # Get the index of avoided skills
            l_index_avoid = set(rng.choice(list(Sa), n_avoid, replace=False))
            
            # For all wanted and avoided skills
            for i in l_index_want.union(l_index_avoid):
                # If i is wanted, mark it as 1.
                if i in l_index_want:
                    l_skillPreferences[i] = 1
                # If i is avoided, mark it as -1.
                else:
                    l_skillPreferences[i] = -1
            
            # Update the avoid/want dictionary of the learner 
            resultDict[l] = [[k,v] for k, v in l_skillPreferences.items()]
        
        # Return the results for all learners
        return resultDict

    def train_test_split(self, learners:dict[str,Any], trainProportion:int = 0.7, seed:int = 0) -> tuple[dict, dict]:
        """Function to separate the learners dataset into a training and a test set

        Args:
            learners (dict[str,Any]): The full learners set
            trainProportion (int, optional): The proportion that will be used in the training set. Defaults to 0.7.
            seed (int, optional): The seed for the choice of the set. Defaults to 0.

        Returns:
            tuple[dict, dict]: The training set and the test set
        """
        # Initialise the rng and get the set of learners
        rng = np.random.RandomState(seed)
        L = list(learners.keys())

        # Get the training set size and sampling index
        n_training = (trainProportion*len(L)).__floor__()
        training_index = list(rng.choice(L, size=n_training, replace=False))
        test_index = [l for l in L if l not in training_index]
        
        # Get the training and test set
        training_set = {l:learners[l] for l in training_index}
        test_set = {l:learners[l] for l in test_index}
        
        # Return the training set and the test set
        return training_set, test_set     

class simpleFuzzyIIFuzzifier():
    def __init__(self, levels:dict[str, float], unknownDefaults:dict[str,Any]|None):
        """A second order fuzzifier that transform Crisp or Fuzzy 1 sets to Fuzzy 2 sets.

        Args:
            levels (dict[str, float]): The default fuzzy 1 level (between 0 and 1) assigned to "beginner", "intermediate" and "expert". It was initially fixed to 0.2, 0.5 and 0.8.
            unknownDefaults (dict[str,Any] | None): What values to put for unknown values. It must contains the information for each datatypes: "data/courses.json", "data/resumes.json" and "data/jobs.json". Moreover, it must contains the `left_provider_confidence` and `right_provider_confidence` dictionaries under the `confidence` dict for "data/resumes.json" and "data/courses.json", but also the `required_confidence` for "data/courses.json" and "data/jobs.json".
        """
        self.unknownDefaults = unknownDefaults
        self.levels = levels

    def fuzzify(self, toFuzzify:dict[str,Any], srcFile:str, userDefault:dict[str,Any]|None = None, **kwargs) -> dict[str,Any]:
        """Function built to fuzzify recursively a skill-expertise set into a type II fuzzy set. As entry, it get a graded skill-expertise set and return a Type II fuzzy set containing Triangles for declared expertise and Ramps for required expertise.
        
        Args:
            toFuzzify (dict[str,Any]): The graded fuzzy set to fuzzify.
            srcFile (str): The source file type being fuzzified between `data/resumes.json`, `data/courses.json` and `data/jobs.json`.
            userDefault (dict[str,Any] | None, optional): The alternative parameters to use for fuzzification. Defaults to None.

        Returns:
            dict[str,Any]: The type II fuzzy sets
            
        #### Example of userDefault:
            {"expertise":0.5, "left_provider_confidence":0.2247808, "right_provider_confidence":0.2247808, "required_confidence":0}
            
            Note that the requirements are expressed in term of dispersion here for the sake of simplicity, but is stored as absolute position.  
        """
        # If there is a dict, traverse the dict recursively
        if isinstance(toFuzzify, dict):
            return {k:self.fuzzify(v, srcFile, userDefault, key=k) for k,v in toFuzzify.items()}
        # Same for the list
        elif isinstance(toFuzzify, list):
            return [self.fuzzify(v, srcFile, userDefault, **kwargs) for v in toFuzzify]
        # Transform the labelled values into fuzzy values.
        elif isinstance(toFuzzify, str): 
            
            # If the user does not ask for additional parameter, use default parameter for unknown. If the value is known, continue in this case no matter what          
            if userDefault is None or self.levels.get(toFuzzify, None) is not None:
                # Get the expertise if it is known or from userDefault/unknownDefaults if it is not 
                expertise = self.levels.get(toFuzzify, self.unknownDefaults[srcFile])
                
                # Same for left confidence, right confidence and required confidence.
                cl = self.unknownDefaults["confidence"]["left_provider_confidence"].get(srcFile, None) if userDefault is None else userDefault["left_provider_confidence"]
                cr = self.unknownDefaults["confidence"]["right_provider_confidence"].get(srcFile, None) if userDefault is None else userDefault["right_provider_confidence"]
                re = self.unknownDefaults["confidence"]["required_confidence"].get(srcFile, None) if userDefault is None else userDefault["required_confidence"]
            
            # If the value is unknown, but we want to use custom parameters 
            else:
                # Get expertise and confidence parameters
                expertise = userDefault["expertise"]
                cl = userDefault["left_provider_confidence"]
                cr = userDefault["right_provider_confidence"]
                re = userDefault["required_confidence"]
            
            # If it is courses requirements, return a ramp
            if srcFile == "data/courses.json" and kwargs["key"]=="required":
                return [expertise, round(max(0, expertise-re), 6)]
            
            # If it is courses acquirements, return a triangle
            elif srcFile == "data/courses.json" and kwargs["key"]=="to_acquire":
                return [expertise, round(expertise-max(0, expertise-cl), 6), round(min(1, expertise+cr)-expertise, 6)]

            # Resume -> triangles
            elif srcFile == "data/resumes.json":
                return [expertise, round(expertise-max(0, expertise-cl), 6), round(min(1, expertise+cr)-expertise, 6)]

            # Jobs -> Requirements
            elif srcFile == "data/jobs.json":
                return [expertise, round(max(0, expertise-re), 6)]
        
        # If the value is fuzzified, return it
        else:
            return toFuzzify   
    
    def getExpertises(self, l:list[list[int,float]]) -> list[float]:
        """Get the registered expertise from a list of skill-expertise pairs.

        Args:
            l (list[list[int,float]]): The list of skill-expertise pair.

        Returns:
            list[float]: The list of expertise in the same order as the skill-expertise pair
        """
        return [pair[1] for pair in l]
    
    def refuzzifyWithLambda(self, toReFuzzify:dict[str, Any], srcFile:str, left_function:Callable[[list[list[int,float]]], list[Any]]|float = 0., right_function:Callable[[list[list[int,float]]], list[Any]]|float = 0., required_function:Callable[[list[list[int,float]]], list[Any]]|float = None) -> dict[str,Any]:
        """Function to transform Fuzzy I sets into Fuzzy II sets using lambda functions to create triangles and ramp.

        Args:
            toReFuzzify (dict[str, Any]): A fuzzy I skill-expertise set.
            srcFile (str): The type of document being fuzzified between `data/courses.json`, `data/resumes.json` and `data/jobs.json`
            left_function (Callable[[list[list[int,float]]], list[Any]] | float, optional): A function that take a list of skill-expertise pair as entry and return a list of left confidence in the same order. If the value is a float, then the confidence is constant . Defaults to 0..
            right_function (Callable[[list[list[int,float]]], list[Any]] | float, optional): Same as left confidence but for the right. Defaults to 0..
            required_function (Callable[[list[list[int,float]]], list[Any]] | float, optional): Same as left confidence, but for the required. Note that if the value is None, then the ramp is identity to the expertise. Defaults to None.

        Returns:
            dict[str,Any]: The type II fuzzy set for the give type I fuzzy set and parameters.
        """
        # Initialise the identity function for required if it is not defined
        if required_function is None:
            required_function = self.getExpertises
        
        # subfunction to create triangular from a list
        def fuzzifyTriangular(value:list) -> list:
            # Get the skills and expertises
            skills = [pair[0] for pair in value]
            expertise = self.getExpertises(value)
            
            # Apply the given function for left and right confidence, or constant
            left_confidence = left_function(value) if isinstance(left_function, Callable) else list(np.full(len(value), left_function))
            right_confidence = right_function(value) if isinstance(right_function, Callable) else list(np.full(len(value), right_function))
            
            # Apply a transformation to retrieve the right format [[id, [e, cl,cr]], ...]
            return list(map(lambda l: list(l), zip(skills, map(lambda l: list(l), zip(expertise, left_confidence, right_confidence)))))
        
        # subfunction to create ramps from a list
        def fuzzifyRamp(value:list) -> list:
            # Get the skills and expertises
            skills = [pair[0] for pair in value]
            expertise = self.getExpertises(value)
            
            # Apply the required function or constant/identity
            required_confidence = required_function(value) if isinstance(required_function, Callable) else list(np.full(len(value), required_function))
            
            # Apply a transformation to retrieve the right format [[id, [e, c]]]
            return list(map(lambda l: list(l), zip(skills, map(lambda l: list(l), zip(expertise, required_confidence)))))

        # Initialise the dictionary to return 
        toReturn = {}
        
        # For each entity
        for key, value in toReFuzzify.items():
            # If it is not a course 
            if isinstance(value, list):
                # Fuzzify into ramp or triangular
                if srcFile == "data/courses.json" or srcFile == "data/resumes.json":
                    toReturn[key] = fuzzifyTriangular(value)
                elif srcFile == "data/jobs.json":
                    toReturn[key] = fuzzifyRamp(value)
            # If it is a course, fuzzify accordingly
            elif isinstance(value, dict):
                toReturn[key]["to_acquire"] = fuzzifyTriangular(toReFuzzify[key]["to_acquire"])
                toReturn[key]["required"] = fuzzifyRamp(toReFuzzify[key]["required"])
        
        # Return the set.
        return toReturn

class NeighbourResumeFuzzyIIFuzzifier(NeighbourResumeFuzzifier):
    def __init__(self, levels:dict[str,float], df_taxonomy:pd.DataFrame, lvlCols:list[str], skillIdCol:str="unique_id"):
        super().__init__(levels, df_taxonomy, lvlCols, skillIdCol)
        
        # Compile the different fuzzification functions
        self.mode = {"associationRules":self.getARDictAlpha}
    
    def getARDictAlpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], frequencyThreshold:int = 10):
        # Get the skills in the known set and retrieve the id of known skills
        skills = [list(i) for i in knownSet]
        skillKnownID = {k[0] for k in knownSet}
                
        # Infer unknown if any
        if unknownSet != set():
            # For each unknown, match the highest rule in term of expertise (if the threshold is passed)
            for unkID, _ in unknownSet:
                possibilities = {self.associationDict[knownID][unkID] for knownID in skillKnownID if self.associationFreqDict[knownID][unkID] >= frequencyThreshold}
                possibilities.difference_update({None})
                skills.append([unkID, None] if len(possibilities) == 0 else [unkID, max(possibilities)])
        
        skills_to_return = []
        for skill, expertise in skills:
            if expertise is None:
                skills_to_return.append([skill, [expertise, None, None]])
                continue
            possibilities = np.array([
                self.associationDict[knownID][skill] 
                for knownID in skillKnownID 
                if self.associationFreqDict[knownID][skill] >= frequencyThreshold 
                and self.associationDict[knownID][skill] is not None
            ] + [expertise])
            
            sub = possibilities[possibilities <= expertise]
            left_confidence = abs(expertise - sub.mean()).__round__(6)   
            
            upp = possibilities[possibilities >= expertise]
            right_confidence = abs(expertise - upp.mean()).__round__(6)

            skills_to_return.append([skill, [expertise, left_confidence, right_confidence]])
        
        # Return the sorted version
        return sorted(skills_to_return, key = lambda x: x[0])

def loadMasteryLevels() -> tuple[dict]:
    """Load the mastery levels. If you have some changes, do it there.

    Returns:
        tuple[dict]: [UPDATE IF IT CHANGE] First the initial fuzzy mastery levels alt <1/3,2/3,1>
    """
            
    # Load alternative mastery level
    with open("fuzzifiedData/fuzzy_mastery_levels_alt.json") as file:
        fuzzyMasteryLevels = json.load(file)
    
    return fuzzyMasteryLevels

def loadResumes() -> tuple[dict]:
    """Load the resumes. If you have some changes, do it there.

    Returns:
        tuple[dict]: [UPDATE IF IT CHANGE] Resumes, then training resumes, then evaluation resumes and debug resumes
    """
    # Load debug resumes
    with open("data/resumes_debug.json", "r") as file:
        debug_resumes = json.load(file)
    
    # Load resumes
    with open("data/resumes.json", "r") as file:
        resumes = json.load(file)        
    
    try:
        # Load training resumes
        with open("data/resumes_train.json", "r") as file:
            training_resumes = json.load(file)
        
        # Load test resumes
        with open("data/resumes_eval.json", "r") as file:
            eval_resumes = json.load(file)
    except:
        training_resumes = None
        eval_resumes = None 
    
    return resumes, training_resumes, eval_resumes, debug_resumes

def loadJobs() -> tuple[dict]:
    """Load the jobs. If you have some changes, do it there.
    
    Returns:
        tuple[dict]: [UPDATE IF IT CHANGE] Load the jobs.
    """
    # Load job positions
    with open("data/jobs.json", "r") as file:
        jobs = json.load(file)
    
    return jobs

def loadCourses() -> tuple[dict]:
    """Load the courses. If you have some changes, do it there.
    
    Returns:
        tuple[dict]: [UPDATE IF IT CHANGE] Load the courses.
    """
    # Load courses
    with open("data/courses.json", "r") as file:
        courses = json.load(file)
        
    return courses

def generateTrainTestResumes_and_save(learners:dict[str,Any], fuzzifier:NeighbourResumeFuzzifier, train_proportion:float = 0.7, seed:int = 0) -> tuple[dict]:
    """Function to generate the train test split and save them in JSON.

    Args:
        learners (dict[str,Any]): The resumes dataset
        train_proportion (float, optional): The proportion that will be used for training. Defaults to 0.7.
        seed (int, optional): The seed of the split. Defaults to 0.

    Returns:
        tuple[dict]: The training and the test set.
    """
    training_resumes, eval_resumes = fuzzifier.train_test_split(learners=learners, trainProportion=train_proportion, seed=seed)
    with open("data/resumes_train.json", "w") as file:
        json.dump(training_resumes, file, indent=4)
    with open("data/resumes_eval.json", "w") as file:
        json.dump(eval_resumes, file, indent=4)
    
    return training_resumes, eval_resumes

def generatePreferences_and_save(resumes:dict[str,Any], fuzzifier:NeighbourResumeFuzzifier, minWant:int = 1, maxWant:int = 5, avoidDistribution:dict[int,float] = {0:0.6,1:0.25,2:0.15}, seed:int = 42):
    """Generate the preference of each users.

    Args:
        resumes (dict[str,Any]): The resumes.
        minWant (int, optional): The minimum number of skill ha. Defaults to 1.
        maxWant (int, optional): _description_. Defaults to 15.
        avoidDistribution (_type_, optional): _description_. Defaults to {0:0.6,1:0.25,2:0.15}.
        seed (int, optional): _description_. Defaults to 42.
    """
    # Preferences on level four of the taxonomy
    with open("data/preferences.json", "w") as file:
        json.dump(
            fuzzifier.sample_want_avoid(resumes, minWant=minWant, maxWant=maxWant, avoidDistribution=avoidDistribution, seed=seed),
            file,
            indent=4
        )
        
    # Preference on level 3
    with open("data/preferencesLV3.json", "w") as file:
        json.dump(
            fuzzifier.sample_want_avoid(resumes, minWant=minWant, maxWant=maxWant, avoidDistribution=avoidDistribution, seed=seed, useLevel3=True),
            file,
            indent=4
        )

def fuzzifyResumesAR(fuzzifierAR:NeighbourResumeFuzzifier, fixedFuzzifier:SimpleFuzzifier, training_resumes:dict[str,Any], eval_resumes:dict[str,Any], association:str, fuzzyMasteryLevels:dict[str,Any], frequencyThreshold:int=1, rootPath:str="fuzzifiedData", prefix:str="alt_"):
    """Function to generate fuzzified training and test set and save them as json. It support all association with fixed, linear, log, gamma and 0 post fuzzification.

    Args:
        fuzzifierAR (NeighbourResumeFuzzifier): The fuzzifier using association rules. Be sure that this fuzzifier has already its association rules computed on the training set. 
        fixedFuzzifier (SimpleFuzzifier): The fuzzifier to apply fixed post fuzzification (For 0 and fixed).
        training_resumes (dict[str,Any]): The training set.
        eval_resumes (dict[str,Any]): The evaluation set
        association (str): One of the supported association `min`, `weighted` or `second`.
        fuzzyMasteryLevels (dict[str,Any]): The mastery levels associated to `beginner`, `intermediate` and `expert` between 0 and 1.
        frequencyThreshold (int, optional): The minimum threshold for which association rules can be used. Defaults to 1.
        rootPath (str, optional): The folder in which the results are stored. Defaults to "fuzzifiedData".
        prefix (str, optional): A prefix to "customize" the files name. Defaults to "alt_".
    """
    # Method used for the post association rule fuzzification
    methods = {
        "Gamma1":{"mode":"weightedLog2","gamma":1}, 
        "0":{"unknownDefault":0}, 
        "Fixed":{"unknownDefault":fuzzyMasteryLevels["beginner"]}, 
        "Linear":{"mode":"linear"}, 
        "Log":{"mode":"log2"}
    }
    
    # Type of dataset used  
    datasets = {
        "train":training_resumes,
        "eval":eval_resumes
    }
    
    # For each methods
    for key, value in methods.items():
        # For each sets
        for datasetMode, resumes in datasets.items():
            # Open the corresponding file 
            with open(f"{rootPath}/{datasetMode}/{prefix}{association}{key}c{frequencyThreshold}_fuzzy_{datasetMode}Resumes.json", "w") as file:
                # Fuzzify using association rules, without loading the AR matrix
                fuzzyARresumes = fuzzifierAR.fuzzify(resumes, mode="associationRules", association=association, frequencyThreshold=frequencyThreshold, loadARMatrix=False)
                
                # Post fuzzify with fixed if it is fixed or 0, with taxonomy methods otherwise
                if key not in {"Fixed", "0"}:
                    fuzzifiedResume = fuzzifierAR.fuzzify(fuzzyARresumes, **value)
                else:
                    fuzzifiedResume = fixedFuzzifier.fuzzify(fuzzyARresumes, **value)
                
                # Dump the files
                json.dump(fuzzifiedResume, file, indent=4)

def fuzzifyResumes(taxonomyFuzzifier:NeighbourResumeFuzzifier, fixedFuzzifier:SimpleFuzzifier, training_resumes:dict[str,Any], eval_resumes:dict[str,Any], fuzzyMasteryLevels:dict[str,Any], rootPath:str="fuzzifiedData", prefix:str="alt_"):
    # Methods for the fuzzification
    methods = {
        "gamma1":{"mode":"weightedLog2","gamma":1},  
        "fixed":{"unknownDefault":fuzzyMasteryLevels["beginner"]}, 
        "linear":{"mode":"linear"}, 
        "log":{"mode":"log2"}
    }
    
    # Training and evaluation set
    datasets = {
        "train":training_resumes,
        "eval":eval_resumes
    }
    
    # For each methods
    for key, value in methods.items():
        # For each sets
        for datasetMode, resumes in datasets.items():
            # Open the corresponding file
            with open(f"{rootPath}/{datasetMode}/{prefix}{key}_fuzzy_{datasetMode}Resumes.json", "w") as file:
                # Fuzzify
                if key not in {"fixed"}:
                    fuzzifiedResume = taxonomyFuzzifier.fuzzify(resumes, **value)
                else:
                    fuzzifiedResume = fixedFuzzifier.fuzzify(resumes, **value)
                
                # Dump the result
                json.dump(fuzzifiedResume, file, indent=4)

if __name__ == "__main__":
    # Load the data
    fuzzyMasteryLevelsAlt = loadMasteryLevels()
    resumes, training_resumes, eval_resumes, debug_resumes = loadResumes()
    jobs = loadJobs()
    courses = loadCourses()
    
    # Instantiate simple fuzzifiers
    ## Fixed fuzzifier (job, courses, resumes)
    fixedFuzzifier = SimpleFuzzifier(fuzzyMasteryLevelsAlt)
    ## Taxonomy fuzzifier (resumes)
    taxonomyFuzzifier = NeighbourResumeFuzzifier(fuzzyMasteryLevelsAlt, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
    
    # Transform the data
    ## Train-Test Split (no need to uncomment if they are already generated)
    training_resumes, eval_resumes = generateTrainTestResumes_and_save(resumes, taxonomyFuzzifier, train_proportion=0.7, seed=0)
    
    ## Generate the preferences
    generatePreferences_and_save(resumes, taxonomyFuzzifier, minWant = 1, maxWant = 5, avoidDistribution = {0:0.6,1:0.25,2:0.15}, seed = 42)
    
    
    ## Association rule fuzzifier (resumes)
    ### Instanciation
    weightedARFuzzifier = NeighbourResumeFuzzifier(fuzzyMasteryLevelsAlt, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
    secondARFuzzifier = NeighbourResumeFuzzifier(fuzzyMasteryLevelsAlt, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
    minARFuzzifier = NeighbourResumeFuzzifier(fuzzyMasteryLevelsAlt, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
        
    ### Association rules fitting
    weightedARFuzzifier.loadAssociationRuleDict(training_resumes, association="weighted")
    secondARFuzzifier.loadAssociationRuleDict(training_resumes, association="second")
    minARFuzzifier.loadAssociationRuleDict(training_resumes, association="min")
        
    # Fit the association rules for Fuzzy II fuzzifier
    #secondARFuzzyIIFuzzifier = NeighbourResumeFuzzyIIFuzzifier(fuzzyMasteryLevelsAlt, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
    #secondARFuzzyIIFuzzifier.loadAssociationRuleDict(debug_resumes, association="second")
    #secondARFuzzyIIFuzzifier.fuzzify(debug_resumes, "associationRules", loadARMatrix=False, frequencyThreshold=1)
    
    
    
    
    #### Fuzzy I with weighted associations
    fuzzifyResumesAR(
        fuzzifierAR=weightedARFuzzifier, 
        fixedFuzzifier=fixedFuzzifier, 
        training_resumes=training_resumes, 
        eval_resumes=eval_resumes, 
        association="weighted", 
        fuzzyMasteryLevels=fuzzyMasteryLevelsAlt,
        frequencyThreshold=1,
        rootPath="fuzzifiedData",
        prefix="alt_"
    )
    
    #### Fuzzy I with second association
    fuzzifyResumesAR(
        fuzzifierAR=secondARFuzzifier, 
        fixedFuzzifier=fixedFuzzifier, 
        training_resumes=training_resumes, 
        eval_resumes=eval_resumes, 
        association="second", 
        fuzzyMasteryLevels=fuzzyMasteryLevelsAlt,
        frequencyThreshold=1,
        rootPath="fuzzifiedData",
        prefix="alt_"
    )

    #### Fuzzy I with min association
    fuzzifyResumesAR(
        fuzzifierAR=minARFuzzifier, 
        fixedFuzzifier=fixedFuzzifier, 
        training_resumes=training_resumes, 
        eval_resumes=eval_resumes, 
        association="min", 
        fuzzyMasteryLevels=fuzzyMasteryLevelsAlt,
        frequencyThreshold=1,
        rootPath="fuzzifiedData",
        prefix="alt_"
    )
    
    #### Fuzzy I based on the taxonomy.
    fuzzifyResumes(
        taxonomyFuzzifier=taxonomyFuzzifier,
        fixedFuzzifier=fixedFuzzifier,
        training_resumes=training_resumes,
        eval_resumes=eval_resumes,
        fuzzyMasteryLevels=fuzzyMasteryLevelsAlt,
        prefix="alt_"
    )
        
    #### On the courses
    with open("fuzzifiedData/alt_fixed_fuzzy_courses.json", "w") as file:
        json.dump(fixedFuzzifier.fuzzify(courses, unknownDefault=UNKNOWN_DEFAULT["data/courses.json"]), file, indent=4)
    
    #### On the jobs
    with open("fuzzifiedData/alt_fixed_fuzzy_jobs.json", "w") as file:
        json.dump(fixedFuzzifier.fuzzify(jobs, unknownDefault=UNKNOWN_DEFAULT["data/jobs.json"]), file, indent=4)
    
    #### On all the resumes
    with open("fuzzifiedData/alt_fixed_fuzzy_resumes.json", "w") as file:
        json.dump(fixedFuzzifier.fuzzify(resumes, unknownDefault=UNKNOWN_DEFAULT["data/resumes.json"]), file, indent=4)
        