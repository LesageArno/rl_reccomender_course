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
    "data/courses.json": 0.5,
    "data/jobs.json": 0.8,
    "data/resumes.json": 0.5,
    "confidence":{
        # Provider
        "left_provider_confidence":{
            "data/courses.json":0.1, 
            "data/resumes.json":0.1
        },
        "right_provider_confidence":{
            "data/courses.json":0.1,
            "data/resumes.json":0.1
        },
        # Require (Warning, contrary to the rest, here, it is the dispersion)
        "required_confidence":{
            "data/courses.json":0,
            "data/jobs.json":0
        }
    }
}

CRISP_DEFAULT = {
    "data/courses.json": 0.5,
    "data/jobs.json": 0.8,
    "data/resumes.json": 0.5,
    "confidence":{
        # Provider
        "left_provider_confidence":{
            "data/courses.json":0, 
            "data/resumes.json":0
        },
        "right_provider_confidence":{
            "data/courses.json":0,
            "data/resumes.json":0
        },
        # Require (Warning, contrary to the rest, here, it is the dispersion)
        "required_confidence":{
            "data/courses.json":0,
            "data/jobs.json":0
        }
    }
}

FuzzificationMethod = Literal["linear","weighted","log2","weightedLog2","associationRules"]
OnTaxonomyFuzzificationMethod = Literal["linear","weighted","log2","weightedLog2"]
AssociationsMethod = Literal["min","weighted","crisp"]

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
        elif isinstance(toFuzzify, str):
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
        self.df_taxonomy = df_taxonomy.copy()
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
            skills.append([skill[0], max(minUnknownLevel,round(alpha,4))])
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
            skills.append([skill[0], max(minUnknownLevel, round(alpha, 4))])
        
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
                alpha = len(neighbourSet.intersection(knownIDSet))*0.5/max(1,np.log2(len(neighbourSet)))
            
            # If the skill is known, keep it as is.
            elif skill in knownSet:
                alpha = skill[1]
            
            # Register the skills
            skills.append([skill[0], max(minUnknownLevel, round(alpha,4))])
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
            alpha = gamma * sum(knownIDtoLevelMap.get(i,0) for i in knownNeighbourSet)/max(1,len(knownNeighbourSet)) + (1-gamma) * (len(knownNeighbourSet))*0.5/max(1,np.log2(len(neighbourSet)))
            skills.append([skill[0], max(minUnknownLevel,round(alpha, 4))])
        
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
                skills.append([
                    unkID,
                    max({
                        self.associationDict[knownID][unkID]
                        for knownID in skillKnownID
                        if self.associationFreqDict[knownID][unkID] >= frequencyThreshold},
                        default=None
                    )             
                ])
        
        # Return the sorted version
        return sorted(skills, key = lambda x: x[0])

    def loadAssociationRuleDict(self, documents:dict[str,Any], association:AssociationsMethod = "weighted") -> None:
        """Generate an Association Rule matrix from the given document (in sparse format). In the end, you obtain a matrix displaying the number of occurrence of a given rule as well as another giving the expertise level we can expect from the association rule A[s1,s2] : s1 -> s2. 

        Args:
            documents (dict[str,Any]): The documents on which we will infer the association rule matrix.
            association (AssociationsMethod): The association method can be `crisp` in this case, whenever we see (s_i,e_i) -> (s_j, e_j) we register at A[i,j]: e_j. It can also be `weighted`: (s_i,e_i)->(s_j,e_j) => A[i,j] = e_i*e_j or `min`: (s_i,e_i)->(s_j,e_j) => A[i,j] = min(e_i,e_j) which corresponds to a fuzzy and. Defaults to "weighted"
        """
        # Create a sparse representation of the association rule matrix
        associationDict = defaultdict(lambda:defaultdict(lambda:0))
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
                    associationFreqDict[s1][s2] += 0 if associationDict[s1][s2] == 0 else 1
                    
                    # Apply the desired association
                    if association == "weighted": 
                        associationDict[s1][s2] += e1*e2
                    elif association == "crisp": 
                        associationDict[s1][s2] += e2
                    elif association == "min": 
                        associationDict[s1][s2] += min(e1, e2)
        
        # Make the average over all keys in the dictionary 
        for k1, v1 in associationDict.items():
            for k2 in v1.keys():
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
        kwargs.pop("loadARMatrix")
        
        # For each individuals
        for person in toFuzzify.keys():
            # Get the skills
            unknownSet, knownSet = self.getSkillsSets(toFuzzify, person)
            
            # Fuzzify the skills levels
            toFuzzify[person] = self.mode[mode](unknownSet, knownSet, **kwargs)
        
        return toFuzzify

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
        
if __name__ == "__main__":
    # Get the fuzzy mastery levels
    with open("fuzzifiedData/fuzzy_mastery_levels.json") as file:
        fuzzyMasteryLevels = json.load(file)    
    
    # Instantiate the fuzzifier
    ## For the Training and Jobs
    fuzzy = SimpleFuzzifier(fuzzyMasteryLevels)
    
    ## For the resumes
    fuzzyNeighbour = NeighbourResumeFuzzifier(fuzzyMasteryLevels, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
    
    ## For Fuzzy II
    fuzzyII = simpleFuzzyIIFuzzifier(fuzzyMasteryLevels, UNKNOWN_DEFAULT)
    fuzzyII_crisp = simpleFuzzyIIFuzzifier(fuzzyMasteryLevels, CRISP_DEFAULT)
    
    # Load resumes
    with open("data/resumes.json", "r") as file:
        resumes = json.load(file)        
    
    ## Fuzzify with default values
    #with open("fuzzifiedData/fuzzy_resumes.json", "w") as file:
    #    json.dump(
    #        fuzzy.fuzzify(resumes, unknownDefault=UNKNOWN_DEFAULT["data/resumes.json"]),
    #        file, 
    #        indent=4
    #    )
    
    
    ## Fuzzify with weighted association rules (occurrences>=1), then Gamma 1 
    #with open("fuzzifiedData/weightedGamma1_fuzzy_resumes.json", "w") as file:
    #    json.dump(
    #        fuzzyNeighbour.fuzzify(fuzzyNeighbour.fuzzify(
    #            resumes, mode="associationRules", weighted=False, association="weighted", frequencyThreshold=1        
    #        ), mode="weightedLog2", gamma=1),
    #        file,
    #        indent=4
    #    )
    
    ## Fuzzify with min association rules (occurrences>=1), then Gamma 1
    #with open("fuzzifiedData/minGamma1_fuzzy_resumes.json", "w") as file:
    #    json.dump(
    #        fuzzyNeighbour.fuzzify(fuzzyNeighbour.fuzzify(
    #            resumes, mode="associationRules", weighted=False, association="min", frequencyThreshold=1
    #        ), mode="weightedLog2", gamma=1),
    #        file,
    #        indent=4
    #    )
    
    ## Constant 0 Fuzzy II for resumes
    with open("fuzzyIIData/degenerated_fuzzyII_resumes.json", "w") as file:
       json.dump(
           fuzzyII_crisp.fuzzify(
               resumes,
               "data/resumes.json",
               use_default=True
           ),
           file,
           indent=4
       )
    
    ## Constant 0.1 Fuzzy II for resumes
    #with open("fuzzyIIData/fixed_fuzzyII_resumes.json", "w") as file:
    #    json.dump(
    #        fuzzyII.fuzzify(
    #            resumes,
    #            "data/resumes.json",
    #            use_default=True
    #        ),
    #        file,
    #        indent=4
    #    )
    
    ## Fixed RMSE for resumes
    #with open("fuzzyIIData/fixedRMSE_fuzzyII_resumes.json", "w") as file:
    #    json.dump(fuzzyII.fuzzify(
    #        resumes,
    #        "data/resumes.json",
    #        {"expertise":0.5, "left_provider_confidence":0.2247808, "right_provider_confidence":0.2247808, "required_confidence":0}
    #    ),
    #        file, indent=4
    #    )
        
    ## Gamma1 RMSE for resumes
    #with open("fuzzyIIData/gamma1RMSE_fuzzyII_resumes.json", "w") as file:
    #    json.dump(
    #        fuzzyII.refuzzifyWithLambda(
    #            fuzzyNeighbour.fuzzify(
    #                fuzzyNeighbour.fuzzify(resumes, mode="associationRules", association="min", frequencyThreshold=1), 
    #                mode="weightedLog2", gamma=1
    #            ), 
    #            srcFile="data/resumes.json", left_function=0.2081636, right_function=0.2081636, required_function=None,
    #        ),
    #        file, indent=4
    #    )
    
    
    
    # Load job positions
    with open("data/jobs.json", "r") as file:
        jobs = json.load(file)
    
    ## Fuzzify jobs
    #with open("fuzzifiedData/fuzzy_jobs.json", "w") as file:
    #    json.dump(fuzzy.fuzzify(jobs, unknownDefault=UNKNOWN_DEFAULT["data/jobs.json"]), file, indent=4)
    
    ## Fixed Fuzzy II for jobs
    # with open("fuzzyIIData/fixed_fuzzyII_jobs.json", "w") as file:
        # json.dump(fuzzyII.fuzzify(jobs, "data/jobs.json"), file, indent=4)
    
    ## Degenerated Fuzzy II for jobs
    with open("fuzzyIIData/degenerated_fuzzyII_jobs.json", "w") as file:
        json.dump(fuzzyII_crisp.fuzzify(jobs, "data/jobs.json"), file, indent=4)
    
    ## Fixed RMSE for jobs
    # with open("fuzzyIIData/fixedRMSE_fuzzyII_jobs.json", "w") as file:
        # json.dump(fuzzyII.fuzzify(
        #     jobs,
        #     "data/jobs.json",
        #     {"expertise":0.8, "left_provider_confidence":0, "right_provider_confidence":0, "required_confidence":0}
        # ),
        #     file, indent=4
        # )
    
    
    
    
    # Load courses
    with open("data/courses.json", "r") as file:
        courses = json.load(file)
        
    ## Fuzzify courses
    #with open("fuzzifiedData/fuzzy_courses.json", "w") as file:
    #    json.dump(fuzzy.fuzzify(courses, unknownDefault=UNKNOWN_DEFAULT["data/courses.json"]), file, indent=4)
    
    ## Fixed Fuzzy II for courses
    # with open("fuzzyIIData/fixed_fuzzyII_courses.json", "w") as file:
    #     json.dump(fuzzyII.fuzzify(courses, "data/courses.json"), file, indent=4)
    
    ## Degenerated Fuzzy II for courses
    with open("fuzzyIIData/degenerated_fuzzyII_courses.json", "w") as file:
        json.dump(fuzzyII_crisp.fuzzify(courses, "data/courses.json"), file, indent=4)

    # ## Fixed RMSE for courses
    # with open("fuzzyIIData/fixedRMSE_fuzzyII_courses.json", "w") as file:
    #     json.dump(fuzzyII.fuzzify(
    #         courses,
    #         "data/courses.json",
    #         {"expertise":0.5, "left_provider_confidence":0.238236, "right_provider_confidence":0.238236, "required_confidence":0.2797116}
    #     ),
    #         file, indent=4
    #     )
    