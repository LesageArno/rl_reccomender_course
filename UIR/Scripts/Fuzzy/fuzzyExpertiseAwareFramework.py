from typing import Any, Annotated
from numpy import floating
 

class FuzzySkillExpertiseSet():
    def __init__(self, skillExpertiseSet:list[list[Any,float]]|dict[Any,float]|list[float], *, fromSelf:bool=False, roundingDecimal:int=6):
        """A class to represent and perform operations on Fuzzy Skill Expertise Set. It can take as input a list of skill-expertise pair, or a dictionary representing directly the skill-expertise set.

        Args:
            skillExpertiseSet (list[list[Any,float]] | dict[Any,float] | list[float]): A list containing pairs of skill/expertise, a dictionary mapping skills to expertise levels or a list containing float where all the float represent the expertise for the skill at index i.
            fromSelf (bool, optional): ~~This is not meant to be used by the user directly. It is used internally to create copies of the object. Defaults to False.~~
            roundingDecimal (int, optional): Operations on floats may result in awkward results (eg. 3.00000000004). As a consequence, we prune the obtained values for such an operation to the defined rounding decimal. Defaults to 6.
        """
        # If the input is a list, transform it into a skill-expertise set.
        if not fromSelf and not isinstance(skillExpertiseSet, dict):
            if isinstance(skillExpertiseSet[0], (float,floating)):
                self.fuzzySkillExpertiseSet:dict[Any, float] = dict(enumerate(skillExpertiseSet))
            else:
                self.fuzzySkillExpertiseSet:dict[Any, float] = {val[0]:val[1] for val in skillExpertiseSet}
        
        # If it already is a skill expertise-set format, register it directly
        else:
            self.fuzzySkillExpertiseSet:dict[Any, float] = skillExpertiseSet
        
        # Register the defined rounding decimal
        self.roundingDecimal = roundingDecimal
        
    def __repr__(self) -> str:
        """Use, as a representation of the fuzzy skill-expertise set, the representation of the dictionary whose encode it. 

        Returns:
            str: The representation
        """
        return self.fuzzySkillExpertiseSet.__repr__()

    def getExpertise(self, skill:Any, default:float = 0) -> float:
        """Get the expertise level of a requested skill.

        Args:
            skill (Any): The skill for which the expertise level is of interest.
            default (float, optional): A default value in case a skill is not in the skill-expertise set. Defaults to 0.

        Returns:
            float: A float representing the expertise level of the given skill.
        """
        return self.fuzzySkillExpertiseSet.get(skill, default)
    
    def Skills(self) -> set[Any]:
        """The set of all skills contained in the skill-expertise set.

        Returns:
            set[Any]: The mentioned set.
        """
        return set(self.fuzzySkillExpertiseSet.keys())
    
    def Levels(self) -> list[float]:
        """The list containing the expertises registered for all skills in the skill-expertise set.

        Returns:
            list[float]: The mentioned list. Note: if you want a set, do not forget to call `set()` over this call.
        """
        return list(self.fuzzySkillExpertiseSet.values())
    
    def Delta(self, other:"FuzzySkillExpertiseSet") -> "FuzzySkillExpertiseSet":
        """A function to compute the expertise gap between two skill-expertise sets. In other words, it returns a skill-expertise set such that all skills that are not in `other` are kept with their corresponding expertise and all skills in both set are kept with the maximum associated expertise.

        Args:
            other (FuzzySkillExpertiseSet): The other set. 

        Returns:
            FuzzySkillExpertiseSet: The new fuzzy skill-expertise set after the gap was computed. The delta is encoded as `Self` \\ `other`
        """
        # Initialise the new skill-expertise set
        fuzzySkillExpertiseDiff = {}
        
        # For each skill the current skill-expertise set possess
        for skill, expertise in self.fuzzySkillExpertiseSet.items():
            # Get the expertise
            fuzzySkillExpertiseDiff[skill] = round(max(0, expertise - other.getExpertise(skill)), self.roundingDecimal)
            
            # If the difference is 0 (skill not in Self or with less expertise than the other set), delete the key
            if fuzzySkillExpertiseDiff[skill] == 0:
                del fuzzySkillExpertiseDiff[skill]
        
        # Return a new FuzzySkillExpertiseSet from the difference and keep the decimal format of the current one
        return FuzzySkillExpertiseSet(fuzzySkillExpertiseDiff, fromSelf=True, roundingDecimal=self.roundingDecimal)
    
    def Union(self, other:"FuzzySkillExpertiseSet") -> "FuzzySkillExpertiseSet":
        """Make the union of two skill-expertise set. The union get all the skills in both set and associate them with the highest expertise level.

        Args:
            other (FuzzySkillExpertiseSet): The other skill-expertise set to make the union with.

        Returns:
            FuzzySkillExpertiseSet: The resulting set of the union of both.
        """
        # Initialise the new expertise set
        fuzzySkillExpertiseUnion = {}
        
        # For all skills in both set
        for skill in self.fuzzySkillExpertiseSet.keys() | other.fuzzySkillExpertiseSet.keys():
            # Associate the skill in the union to the maximum related expertise
            fuzzySkillExpertiseUnion[skill] = max(self.getExpertise(skill), other.getExpertise(skill))
        
        # Create a FuzzySkillExpertiseSet from the obtained union set.
        return FuzzySkillExpertiseSet(fuzzySkillExpertiseUnion, fromSelf=True, roundingDecimal=self.roundingDecimal)
    
    def IsIncludedIn(self, other:"FuzzySkillExpertiseSet") -> bool:
        """Check whether all skills in the current skill-expertise set is included within the other set with an expertise level of at most the one in the other skill-expertise set.

        Args:
            other (FuzzySkillExpertiseSet): The other set to check inclusion with.

        Returns:
            bool: True if Self is included in other, False otherwise.
        """
        # For all the skills
        for skill in self.fuzzySkillExpertiseSet.keys() | other.fuzzySkillExpertiseSet.keys():
            # If an expertise of Self is higher than in other, then it is not included. (Reminder: default expertise is 0 when not in set)
            if self.getExpertise(skill) > other.getExpertise(skill):
                return False
        
        # If all included, then return True
        return True
    
    def __sub__(self, other:"FuzzySkillExpertiseSet") -> "FuzzySkillExpertiseSet":
        """Minus operator overload. It is used as an alias for the Delta operation.

        Args:
            other (FuzzySkillExpertiseSet): The right member of the subtraction.

        Returns:
            FuzzySkillExpertiseSet: The Delta.
        """
        return self.Delta(other)
    
    def __or__(self, other:"FuzzySkillExpertiseSet") -> "FuzzySkillExpertiseSet":
        """An overload of the set union operator. It is used as an alias for the fuzzy skill-expertise union.

        Args:
            other (FuzzySkillExpertiseSet): The right member of the union.

        Returns:
            FuzzySkillExpertiseSet: The Union.
        """
        return self.Union(other)

    def __getitem__(self, skill:Any) -> float:
        """Overload of the Indexing operator. It is used to get the expertise level of a particular skill.

        Args:
            skill (Any): The skill for which the expertise is requested.

        Returns:
            float: The corresponding expertise. Default to 0.
        """
        return self.getExpertise(skill)

    def mergeTrainingIfRequirements(self, training:"Training") -> "FuzzySkillExpertiseSet":
        """Make the union of a training acquisition and the current skill-expertise set iff Self passed the requirements to attend the training.

        Args:
            training (Training): The training which is supposed to be passed.

        Returns:
            FuzzySkillExpertiseSet: A skill-expertise set corresponding to either the union of the training acquired skills and the person skills, or the person skills if the requirements were not met. 
        """
        # Check and apply changes if the requirements are met
        if training[0].IsIncludedIn(self):
            return self.Union(training[1])
        
        # Default to the user skill-expertise set
        return FuzzySkillExpertiseSet(self.fuzzySkillExpertiseSet, fromSelf=True, roundingDecimal=self.roundingDecimal)
    
    def computeUsefulContent(self, goal:"Job", training:"Training") -> float:
        """Compute the useful content of a training to achieve a defined goal.

        Args:
            goal (Job): The goal to achieve. (It is a FuzzySkillExpertiseSet representing the prerequisites of a job).
            training (Training): A training.

        Returns:
            float: A float representing the amount of expertise gained toward a defined goal. The higher the better.
        """
        # Compute the useful content
        return round(
            sum(goal.Delta(self).Levels()) - # Get the skill proficiency gap 
            sum(goal.Delta(self.mergeTrainingIfRequirements(training)).Levels()), # Get the skill proficiency gap after the training
            self.roundingDecimal # Round the result
        )

    def computeAggregatedUsefulContent(self, goals:"Goals", training:"Training", *, use_max:bool=False) -> float:
        """Compute the useful content of a training toward a set of goals.

        Args:
            goals (Goals): A list of Job. It contains the prerequisite to fill an application.
            training (Training): The training component.
            use_max (bool, optional): If True, change the aggregation from a sum to the maximum expertise. Defaults to False.

        Returns:
            float: The sum of useful content for each goals.
        """
        # Compute the useful content for each goals
        usefulContent = list(map(lambda goal: self.computeUsefulContent(goal, training), goals))
        
        # Aggregate the results
        return round(sum(usefulContent), self.roundingDecimal) if not use_max else max(usefulContent)
    
    def computeMissingContent(self, goal:"Job", training:"Training") -> float:
        """Compute the missing content of a training to achieve a defined goal.

        Args:
            goal (Job): The goal to achieve. (It is a FuzzySkillExpertiseSet representing the prerequisites of a job).
            training (Training): A training.

        Returns:
            float: A float representing the amount of expertise missing after the training to achieve the goal goal. The lower the better.
        """
        # Compute the missing content
        return round(
            sum(goal.Delta(self.mergeTrainingIfRequirements(training)).Levels()), # Get the skill-proficiency gap after training
            self.roundingDecimal # Round to the requested decimal
        )
    
    def computeAggregatedMissingContent(self, goals:"Goals", training:"Training", *, use_max:bool=False) -> float:
        """Compute the missing content of a training to a set of goals.

        Args:
            goals (Goals): A list of Job. It contains the prerequisite to fill an application.
            training (Training): The training component.
            use_max (bool, optional): If True, change the aggregation from a sum to the maximum. Defaults to False.

        Returns:
            float: The sum of missing content for each goals.
        """
        # Compute the missing content
        missingContent = list(map(lambda goal: self.computeMissingContent(goal, training), goals))
        
        # Aggregate the results
        return round(sum(missingContent), self.roundingDecimal) if not use_max else max(missingContent)
    
    def computeUnnecessaryContent(self, goal:"Job", training:"Training") -> float:
        """Compute the unnecessary content of a training to achieve a defined goal.

        Args:
            goal (Job): The goal to achieve. (It is a FuzzySkillExpertiseSet representing the prerequisites of a job).
            training (Training): A training.

        Returns:
            float: A float representing the amount of expertise gained outside the targeted goal after training. The lower the better.
        """
        # Compute the unnecessary component
        return round(
            sum(training[1].Delta(goal.Delta(self)).Levels()), # Get the skill-proficiency gap between the training and the skill proficiency gap of a user to a defined goal
            self.roundingDecimal # Round the result
        )
    
    def computeAggregatedUnnecessaryContent(self, goals:"Goals", training:"Training", *, use_max:bool=False) -> float:
        """Compute the unnecessary content of a training to a set of goals.

        Args:
            goals (Goals): A list of Job. It contains the prerequisite to fill an application.
            training (Training): The training component.
            use_max (bool, optional): If True, change the aggregation from a sum to the maximum. Defaults to False.

        Returns:
            float: The sum of unnecessary content for each goals.
        """
        # Compute the unnecessary component
        unnecessaryContent = list(map(lambda goal: self.computeUnnecessaryContent(goal, training), goals))
        
        # Aggregate the results
        return round(sum(unnecessaryContent), self.roundingDecimal) if not use_max else max(unnecessaryContent)
    
    def getCompletedGoalsAfterTraining(self, goals:"Goals", training:"Training") -> "Goals":
        """Get the set of goals that were achieved doing a defined training.

        Args:
            goals (Goals): A set of goals to achieve.
            training (Training): A training

        Returns:
            Goals: A list of Job (Goals) that were achieved by the training
        """
        # Initialise the achieved goals
        completedGoals = []
        
        # For each job
        for goal in goals:
            # If one can be achieved after the training, but not before
            if goal.IsIncludedIn(self.mergeTrainingIfRequirements(training)) and not goal.IsIncludedIn(self):
                # Append the job to the realised goals
                completedGoals.append(FuzzySkillExpertiseSet(goal.fuzzySkillExpertiseSet, fromSelf=True, roundingDecimal=self.roundingDecimal))
        # Return the list of realised goal.
        return completedGoals
    
    def computeUsefulnessDegree(self, goals:"Goals", training:"Training", *, use_max_useful:bool=False, use_max_missing:bool=False, use_max_unnecessary:bool = False) -> float:
        """Compute the usefulness degree of a training for a given set of goals.

        Args:
            goals (Goals): A set of goals (list of jobs).
            training (Training): A training.
            use_max_useful (bool, optional): Change the aggregation from a sum to a maximum in the useful component. Defaults to False.
            use_max_missing (bool, optional): Change the aggregation from a sum to a maximum in the missing component. Defaults to False.
            use_max_unnecessary (bool, optional): Change the aggregation from a sum to a maximum in the unnecessary component. Defaults to False.

        Returns:
            float: The usefulness degree of a given training toward a set of goals.
        """
        # Compute the cardinality of the goals set.
        Gcard = len(goals)
        
        # Get the cardinality of the set of goals completed after training
        Jt = len(self.getCompletedGoalsAfterTraining(goals, training))
        
        # Compute the useful component, the missing component and the unnecessary component
        Cut = self.computeAggregatedUsefulContent(goals, training, use_max=use_max_useful)
        Cmt = self.computeAggregatedMissingContent(goals, training, use_max=use_max_missing)
        Cunt = self.computeAggregatedUnnecessaryContent(goals, training, use_max=use_max_unnecessary)
        
        # Return the usefulness degree of the training
        return round((1/(Gcard+1))*(Jt+Cut/(Cut+Cmt+Cunt/(Cunt+1))), self.roundingDecimal)
    
Training = Annotated[tuple[FuzzySkillExpertiseSet, FuzzySkillExpertiseSet], "The first element is the requirement of the training; the second the acquired skills after training."]
Job = Annotated[FuzzySkillExpertiseSet, "A Skill-Expertise set representing the requirements for a job application."]
Goals = Annotated[list[Job], "A list of job for which the user want to apply."]
CV = Annotated[FuzzySkillExpertiseSet, "A Skill-Expertise set representing the knowledge state of an applicant."]

if __name__ == "__main__":
    
    cv1:CV = CV({"A":0.8,"B":0.4,"C":0.6,"D":0.2,"E":0.3})
    
    t1:Training = Training((
        FuzzySkillExpertiseSet({"A":0.8,"B":0.2,"E":0.1}),
        FuzzySkillExpertiseSet({"F":0.6,"G":0.3,"D":0.5})
    ))
    
    t2:Training = Training((
        FuzzySkillExpertiseSet({}),
        FuzzySkillExpertiseSet({"B":0.3,"C":0.3,"D":0.3})
    ))
    
    t3:Training = Training((
        FuzzySkillExpertiseSet({"C":0.6}),
        FuzzySkillExpertiseSet({"C":0.9,"D":0.8})
    ))
    
    j1:Job = Job({"C":0.9,"F":0.2,"A":0.6})
    j2:Job = Job({"A":0.2,"E":0.2})
    j3:Job = Job({"F":0.4,"G":0.2,"D":0.3})
    
    G = Goals([j1, j2, j3])
