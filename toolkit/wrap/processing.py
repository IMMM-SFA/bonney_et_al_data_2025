import pandas as pd
from pandas import DataFrame

def process_diversion_csv(diversions: DataFrame, column_names=["diversion_or_energy_shortage", "diversion_or_energy_target"], compute_shortage_ratio=True):
    diversions["date"] = pd.to_datetime(diversions[["year", "month"]].assign(DAY=1))
    shortage = diversions.diversion_or_energy_shortage 
    target = diversions.diversion_or_energy_target
    
    data = {}
    for column_name in column_names:
        data[column_name] = diversions.pivot_table(
            index="date",
            columns="water_right_identifier",
            values=column_name,
            dropna=False
        )
    if compute_shortage_ratio:
        diversions["shortage_ratio"] = 1 - ((target - shortage) / target)
        data["shortage_ratio"] = diversions.pivot_table(
            index="date",
            columns="water_right_identifier",
            values="shortage_ratio",
            dropna=False
        )
    return data

def process_reservoir_csv(diversions: DataFrame, column_names):
    diversions["date"] = pd.to_datetime(diversions[["year", "month"]].assign(DAY=1))
    data = {}
    for column_name in column_names:
        data[column_name] = diversions.pivot_table(
            index="date",
            columns="reservoir_identifier",
            values=column_name,
            dropna=False
        )
    return data