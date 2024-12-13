# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:18:45 2024

@author: zshawver
"""

import pandas as pd
from juror import Juror

def prepare_juror_lists(df: pd.DataFrame,dv: str,juror_name: str,plaintiff_label: str,defense_label: str) -> dict:

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
