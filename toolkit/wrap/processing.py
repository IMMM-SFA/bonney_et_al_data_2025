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


def aggregate_over_entities(da, entity_dim, agg, block=100):
    import numpy as np
    n = da.sizes["realization"]
    parts = []
    for start in range(0, n, block):
        chunk = da.isel(realization=slice(start, start + block)).load()
        chunk = chunk.where(np.isfinite(chunk))
        if da.name == "shortage_ratio":
            chunk = chunk.where((chunk >= 0) & (chunk <= 1))
        reduced = chunk.sum(dim=entity_dim, skipna=True) if agg == "sum" else chunk.mean(dim=entity_dim, skipna=True)
        parts.append(reduced.values)
    return np.concatenate(parts, axis=0)
