
import pandas as pd
from juror import Juror

def prepare_juror_lists(df: pd.DataFrame, dv, juror_name: str, plaintiff_label, defense_label) -> dict:

    #Empty dictionary to hold jurors' information
    jurors = {}

    #Add jurors to dictionary, jurorNumber:Juror
    for i, row in df.iterrows():
        jurorNumber = "j{}".format(i+1)
        name = row[juror_name] #get juror's name
        ivDict = row.drop(columns = juror_name).to_dict() #Rest of juror's responses
        jurors[jurorNumber] = Juror(name,ivDict)

    #Separate out plaintiff and defense jurors
    return jurors, {jNum:j for jNum,j in jurors.items() if getattr(j,dv) == plaintiff_label}, {jNum:j for jNum,j in jurors.items() if getattr(j,dv) == defense_label}

def calculate_entropy(row):
    total = row["N"]
    if total == 0:
        return 0
    p_plaintiff = row["n_plaintiff"] / total
    p_defense = row["n_defense"] / total
    if p_plaintiff == 0 or p_defense == 0:  # Avoid log(0)
        return 0
    entropy = -(
        p_plaintiff * np.log(p_plaintiff) + p_defense * np.log(p_defense)
    )
    return entropy
