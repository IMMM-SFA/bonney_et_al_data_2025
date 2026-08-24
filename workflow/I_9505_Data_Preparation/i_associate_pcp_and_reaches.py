"""
This script finds the nearest reach to each control point and saves the results to a shapefile and CSV.
"""

import pandas as pd
import os
from toolkit import outputs_path, repo_data_path
from toolkit.data.filter import load_gage_and_reach_shapefiles, associate_gages_to_reaches

### Settings ###
# None

### Path Configuration ###
reach_paths = [
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline12.shp"),
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline13.shp"),
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline11.shp"),
]
gage_paths = [
    os.path.join(repo_data_path, "geospatial", "wrap_gages", "Primary_CP_colorado.shp"),
    os.path.join(repo_data_path, "geospatial", "wrap_gages", "Primary_CP_sabine.shp"),
    os.path.join(repo_data_path, "geospatial", "wrap_gages", "Primary_CP_trinity.shp"),
]

output_dir = outputs_path / "9505"
output_dir.mkdir(exist_ok=True)

reach_shp_path = output_dir / "reaches_with_associated_gages.shp"

pcp_reach_path = output_dir / "pcp_to_reach_mapping.csv"

### Functions ###
# None

### Main ###

def main():
    # Load shapefiles
    reaches_gdf, gages_gdf = load_gage_and_reach_shapefiles(reach_paths, gage_paths)

    # Find nearest reaches
    results_gdf = associate_gages_to_reaches(gages_gdf, reaches_gdf)

    # Save results
    results_gdf.to_file(reach_shp_path)

    # Create combined dataframe of PCPs and their reaches
    pcp_reach_df = pd.DataFrame({
        'PCP_NAME': gages_gdf.loc[results_gdf['associated_gage_idx'], 'ID'].values,
        'REACH_COMID': results_gdf['COMID'].values,
        'DISTANCE_TO_GAGE_M': results_gdf['distance_to_gage_m'].values,
    })

    # Sort by basin and PCP name
    pcp_reach_df = pcp_reach_df.sort_values(['PCP_NAME'])

    # Save to CSV
    pcp_reach_df.to_csv(pcp_reach_path, index=False)

if __name__ == "__main__":
    main()
