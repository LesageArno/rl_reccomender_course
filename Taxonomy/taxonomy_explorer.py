import pandas as pd
from taxonomy_index import LEVEL_COLS

def getTaxonomyID(df_taxonomy:pd.DataFrame, lvl_cols: list[str], skillID_col:str ="unique_id") -> tuple[dict[int,tuple[str,int]],dict[tuple[str,int],int]]:
    """A function to generate unique id for each skills and families of skills in the given taxonomy. 

    Args:
        df_taxonomy (pd.DataFrame): The taxonomy.
        lvl_cols (list[str]): The columns indicating the levels of the taxonomy.
        skillID_col (str, optional): The column indicating the default id for each last level skill. Defaults to "unique_id".

    Returns:
        tuple[dict[int,tuple[str,int]],dict[tuple[str,int],int]]: Two dictionaries. The first associate the id of a skill (or a family of skills) to its name and level in the taxonomy **[ format: {\\<id\\>:(\\<name\\>,\\<level\\>)} ]**. 
        The second dictionary inverse the keys and the values.    
    """
    
    # Initialise the dict that will contains all the ids and the id shift at 0
    uid2name = {}
    base_uid = 0
    
    # For each given levels (except the last)
    for col_lvl, col in enumerate(lvl_cols[:-1]):
        # For each mentioned skills in the taxonomy at the given level
        for id, name in enumerate(df_taxonomy[col].unique().tolist()):
            # Register a unique id and associate it the name and the level of the skill within the taxonomy
            uid2name[id+base_uid] = (name, col_lvl)
        # Operate the shift to avoid id overlap when changing the levels
        base_uid += id + 1

    # If the unique id of each skills at the last level is bigger than the biggest created id, conserve the original id. In the other case, change the original ids to avoid overlaps
    if max(uid2name.keys()) >= min(df_taxonomy[skillID_col]):
        base_uid = max(uid2name.keys()) + 1
    else:
        base_uid = 0
        
    # Register the id of the last level
    for id in df_taxonomy[skillID_col].unique().tolist():
        uid2name[base_uid + id] = (df_taxonomy.loc[df_taxonomy[skillID_col]==id, lvl_cols[-1]].unique().tolist()[0], len(lvl_cols)-1)
    
    # Reverse the dictionary st an id can be given by the name and the level of the skill
    name2uid = {v:k for k, v in uid2name.items()}
    return uid2name, name2uid

def getTaxonomy(df_taxonomy:pd.DataFrame, name2uid_taxonomy:dict[tuple[str,int],int], lvl_cols: list[str]) -> tuple[dict[int|None,int],dict[int|None,list[int]]]:
    """A function that transform the dataframe taxonomy into a navigable taxonomy.

    Args:
        df_taxonomy (pd.DataFrame): The dataframe containing the taxonomy.
        name2uid_taxonomy (dict[tuple[str,int],int]): The dictionary that map the name (and level) of a skill to its unique id. 
        lvl_cols (list[str]): The columns indicating the levels of the taxonomy.

    Returns:
        tuple[dict[int|None,int],dict[int|None,list[int]]]: Two dictionary containing the Bottom Up and the Top Down taxonomy. The Bottom Up format is the following: **{\\<id\\>:\\<parentID\\>}**. 
        The Top Down taxonomy has the following format **{\\<id\\>:[\\<children1ID\\>,...]}**
    """
    # Initialise the bottom up taxonomy and the top down taxonomy
    bottomUpTaxonomy = {}
    topDownTaxonomy = {}
    
    # For each skill
    for key, value in name2uid_taxonomy.items():
        # Get the parent family/skill name if it exists
        if key[1] != 0:
            higherLvlName = df_taxonomy.loc[df_taxonomy[lvl_cols[key[1]]]==key[0], lvl_cols[max(0,key[1]-1)]].tolist()[0]
        else:
            higherLvlName = "[ROOT]"

        # Associate the id of the skill to the id of its parent or None if it does not have any parent
        bottomUpTaxonomy[value] = name2uid_taxonomy.get((higherLvlName, max(key[1]-1,0)), None)
    
    # For each skill in the bottom up taxonomy, get the children and associate it with its parent
    for key, value in bottomUpTaxonomy.items():
        topDownTaxonomy[value] = topDownTaxonomy.get(value, [])
        topDownTaxonomy[value].append(key)
    
    return bottomUpTaxonomy, topDownTaxonomy

def getParent(bottomUpTaxonomy:dict[int,int], id:int):
    return bottomUpTaxonomy[id]

def getChildren(topDownTaxonomy:dict[int,list[int]], id:int):
    return topDownTaxonomy[id]

def getNeighbours(bottomUpTaxonomy:dict[int,int], topDownTaxonomy:dict[int,list[int]], id:int):
    return topDownTaxonomy[bottomUpTaxonomy[id]]


if __name__ == "__main__":
    df = pd.read_csv("data/taxonomy.csv")
    uid2name_taxonomy, name2uid_taxonomy = getTaxonomyID(df, LEVEL_COLS)
    bottomUpTaxonomy, topDownTaxonomy = getTaxonomy(df, name2uid_taxonomy, LEVEL_COLS)
