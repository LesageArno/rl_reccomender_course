from typing import Any, Literal

from ....Taxonomy import taxonomy_explorer as te
import json
import pandas as pd
import numpy as np

UNKNOWN_DEFAULT = {
    "data/courses.json": 0.5,
    "data/jobs.json": 0.8,
    "data/resumes.json": 0.5
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
            "associationRules":self.getARAlpha
        }
        
        # Matrices for bi-gram fuzzification
        self.associationRuleMatrix = None
        self.associationRuleFrequencies = None 
        
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
    
    def getARAlpha(self, unknownSet:set[tuple[int,None]], knownSet:set[tuple[int,float]], frequencyThreshold:int = 10, weighted:bool = False):
        """Function to guess the real expertise level of unknown levels. It is computed based on the association rules retrieved by the `self.loadAssociationRuleMatrix`.
        In opposition to the other methods, this method might leave unknown if no associations rules were inferred for the unknown values. 

        Args:
            unknownSet (set[tuple[int,None]]): The set of skills with unknown levels.
            knownSet (set[tuple[int,float]]): The set of skills with known levels.
            frequencyThreshold (int, optional): The frequency of association observation from which we start to believe the inferred association rule . Defaults to 10.
            weighted (bool, optional): If weighted, then get the expertise level of known levels skills to weight the maximum value found. Defaults to False.

        Returns:
            list[int, float]: A list of skills with updated levels
        """
        # Initialise the skills
        skills = []
        
        # Get the knownIDSet (set of known level skils)
        knownIDtoLevelMap = {i[0]:i[1] for i in knownSet}
        knownIDList = [i for i in knownIDtoLevelMap.keys()]
        
        # Keep only the association rules above a particular threshold
        freqCorRuleMatrix = self.associationRuleMatrix
        mask = self.associationRuleFrequencies[self.nonZeroARRow, self.nonZeroARCol] < frequencyThreshold
        freqCorRuleMatrix[self.nonZeroARRow[mask], self.nonZeroARCol[mask]] = 0
                
        # For each skills
        for skill in unknownSet.union(knownSet):
            # If it is known, just register it
            if skill in knownSet:
                skills.append([skill[0],skill[1]])
                continue
            
            # Find the local applicable association rules
            localAssociations = freqCorRuleMatrix[knownIDList,skill[0]]
            
            # If the user know no skills, then keep at unknown
            if localAssociations.shape[0] == 0:
                skills.append([skill[0], None])
            
            # If we do not want to weight, then consider the max value of local association as truth value
            elif not weighted:
                # Compute the max and keep at unknown if max is zero
                maxExpertise = np.max(localAssociations) if np.max(localAssociations) != 0 else None
                # Register the skill
                skills.append([skill[0], maxExpertise])
            else:
                # Get the maximum weighted expertise-association with regard to known skills (note: this is np.ndarray*list). Keep at unknown if zero
                maxWeightedExpertise = np.max(localAssociations*[knownIDtoLevelMap[i] for i in knownIDList])
                maxWeightedExpertise = round(maxWeightedExpertise,4) if maxWeightedExpertise != 0 else None 
                
                # Register the skill
                skills.append([skill[0], maxWeightedExpertise])
        
        # Return the updated skills        
        return sorted(skills, key=lambda x: x[0])

    def loadAssociationRuleMatrix(self, documents:dict[str,Any], association:AssociationsMethod = "weighted") -> None:
        """Generate an Association Rule matrix from the given document. In the end, you obtain a matrix displaying the number of occurrence of a given rule as well as another giving the expertise level we can expect from the association rule A[s1,s2] : s1 -> s2. 

        Args:
            documents (dict[str,Any]): The documents on which we will infer the association rule matrix.
            association (AssociationsMethod): The association method can be `crisp` in this case, whenever we see (s_i,e_i) -> (s_j, e_j) we register at A[i,j]: e_j. It can also be `weighted`: (s_i,e_i)->(s_j,e_j) => A[i,j] = e_i*e_j or `min`: (s_i,e_i)->(s_j,e_j) => A[i,j] = min(e_i,e_j) which corresponds to a fuzzy and. Defaults to "weighted"
        """
        # Get the maximum id of the taxonomy
        n = max(self.bottomUpTaxonomy)
        
        # Create the Association Rule matrix and the frequency matrix
        # Interpretation of the Association Rule matrix: If skill2 is unknown, having skill1 means you have a minima on average A[skill1, skill2] expertise level 
        associationRuleMatrix = np.zeros((n+1,n+1), dtype=float)
        count = np.ones_like(associationRuleMatrix, dtype=int)
        
        # Fuzzify to keep the unknown only
        documents = SimpleFuzzifier(self.levels).fuzzify(documents, None)
        
        # For every resume, observe the frequency of skills being together
        for doc in documents.keys():
            # Get the set of skill possessed by this doc
            unknownSet, knownSet = self.getSkillsSets(documents, doc)
            skillSet = unknownSet | knownSet
            
            # For each pair of skills
            for skill1 in skillSet:
                for skill2 in skillSet:
                    # If the pair is equal, pass
                    if skill1 == skill2:
                        continue
                    # If one is unknown, skip, we cannot say anything 
                    if skill1[1] is None or skill2[1] is None:
                        continue
                    
                    # Add to count and make the association rule.
                    count[skill1[0], skill2[0]] += 0 if associationRuleMatrix[skill1[0], skill2[0]]==0 else 1
                    
                    # Register the association rule with respect to a given method
                    if association == "weighted":
                        associationRuleMatrix[skill1[0], skill2[0]] += skill1[1]*skill2[1]
                    elif association == "crisp":
                        associationRuleMatrix[skill1[0], skill2[0]] += skill2[1]
                    elif association == "min":
                        associationRuleMatrix[skill1[0], skill2[0]] += min(skill1[1], skill2[1])
        
        # Make the average         
        associationRulesMatrix = associationRuleMatrix/count
        
        # Save the results
        self.nonZeroARRow, self.nonZeroARCol = np.nonzero(associationRuleMatrix) 
        self.associationRuleFrequencies = count
        self.associationRuleMatrix = associationRulesMatrix.round(4)

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
            self.loadAssociationRuleMatrix(toFuzzify, association = kwargs.pop("association", "weighted"))
        kwargs.pop("loadARMatrix")
        
        # For each individuals
        for person in toFuzzify.keys():
            # Get the skills
            unknownSet, knownSet = self.getSkillsSets(toFuzzify, person)
            
            # Fuzzify the skills levels
            toFuzzify[person] = self.mode[mode](unknownSet, knownSet, **kwargs)
        
        return toFuzzify


if __name__ == "__main__":
    # Get the fuzzy mastery levels
    with open("fuzzifiedData/fuzzy_mastery_levels.json") as file:
        fuzzyMasteryLevels = json.load(file)    
    
    # Instantiate the fuzzifier
    ## For the Training and Jobs
    fuzzy = SimpleFuzzifier(fuzzyMasteryLevels)
    
    ## For the resumes
    fuzzyNeighbour = NeighbourResumeFuzzifier(fuzzyMasteryLevels, pd.read_csv("data/taxonomy.csv"), te.LEVEL_COLS)
    
    # Load resumes
    with open("data/resumes.json", "r") as file:
        resumes = json.load(file)
    
    ## Fuzzify with weighted association rules (occurrences>=1), then Gamma 1 
    with open("fuzzifiedData/weightedGamma1_fuzzy_resumes.json", "w") as file:
        json.dump(
            fuzzyNeighbour.fuzzify(fuzzyNeighbour.fuzzify(
                resumes, mode="associationRules", weighted=False, association="weighted", frequencyThreshold=1        
            ), mode="weightedLog2", gamma=1),
            file,
            indent=4
        )
    ## Fuzzify with min association rules (occurrences>=1), then Gamma 1
    with open("fuzzifiedData/minGamma1_fuzzy_resumes.json", "w") as file:
        json.dump(
            fuzzyNeighbour.fuzzify(fuzzyNeighbour.fuzzify(
                resumes, mode="associationRules", weighted=False, association="min", frequencyThreshold=1
            ), mode="weightedLog2", gamma=1),
            file,
            indent=4
        )
    
    
    # Load job positions
    with open("data/jobs.json", "r") as file:
        jobs = json.load(file)
    ## Fuzzify jobs
    with open("fuzzifiedData/fuzzy_jobs.json", "w") as file:
        json.dump(fuzzy.fuzzify(jobs, unknownDefault=UNKNOWN_DEFAULT["data/jobs.json"]), file, indent=4)
    
    # Load courses
    with open("data/courses.json", "r") as file:
        courses = json.load(file)
    ## Fuzzify courses
    with open("fuzzifiedData/fuzzy_courses.json", "w") as file:
        json.dump(fuzzy.fuzzify(courses, unknownDefault=UNKNOWN_DEFAULT["data/courses.json"]), file, indent=4)
    