"""
This script downloads the raw 9505 data from the HydroSource2 website, scoped to only
the HUC8 subregions that actually contain a basin's reach of interest -- i.e. a reach
some WRAP control point maps to (see i_associate_pcp_and_reaches.py, which must run
first to produce pcp_to_reach_mapping.csv).
"""

import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from toolkit import outputs_path, repo_data_path
from toolkit.data.filter import find_huc8_codes_for_reaches

### Settings ###
url = "https://hydrosource2.ornl.gov/files/SWA9505V3Flow/" # URL to 9505 endpoint

### Path Configuration ###
download_folder = outputs_path / "9505" / "raw" # Folder to download the data to

reach_shp_paths = [
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline11.shp"),
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline12.shp"),
    os.path.join(repo_data_path, "geospatial", "9505_shapefiles", "NHDFlowline13.shp"),
]
pcp_reach_mapping_path = outputs_path / "9505" / "pcp_to_reach_mapping.csv"

### Functions ###
def filter_urls_by_huc8(urls, huc8_codes):
    """Keep only .nc URLs whose embedded HUC8 code is in huc8_codes. The filename's
    HUC8 field is an 8-digit code plus a trailing sub-network letter (e.g.
    "12030107N"), so only the first 8 characters are compared."""
    filtered_urls = []
    for url in urls:

        filename = url.split("/")[-1]
        huc8_code = filename.split("_")[2][:8]

        if huc8_code in huc8_codes:
            filtered_urls.append(url)

    return filtered_urls


### Main ###

def main():
    # Determine which HUC8 subregions actually contain a reach of interest, so we
    # only download those instead of every HUC8 in the basins' broader HUC2 regions.
    if not pcp_reach_mapping_path.exists():
        raise FileNotFoundError(
            f"{pcp_reach_mapping_path} not found. Run i_associate_pcp_and_reaches.py first "
            "to determine which reaches the basins' WRAP control points need."
        )
    pcp_reach_mapping = pd.read_csv(pcp_reach_mapping_path)
    comids = pcp_reach_mapping["REACH_COMID"].astype(int).unique().tolist()
    huc8_codes = find_huc8_codes_for_reaches(reach_shp_paths, comids)
    print(f"Downloading data for {len(huc8_codes)} HUC8 subregions covering {len(comids)} reaches of interest.")

    # Get the page contents
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all subfolder links. Absolute hrefs (e.g. "/files/") are parent-directory
    # links, not ensemble-member subfolders, so they're excluded.
    nc_folders = [
        url + link.get("href") for link in soup.find_all("a")
        if link.get("href").endswith("/") and not link.get("href").startswith("/")
    ]

    # Download each .nc file
    os.makedirs(download_folder, exist_ok=True)

    for folder in nc_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        os.makedirs(os.path.join(download_folder, folder_name), exist_ok=True)
        response = requests.get(folder)
        soup = BeautifulSoup(response.text, "html.parser")
        nc_files = [folder + link.get("href") for link in soup.find_all("a") if link.get("href").endswith(".nc")]
        nc_files = filter_urls_by_huc8(nc_files, huc8_codes)

        for file_url in nc_files:
            filename = os.path.join(download_folder, folder_name, os.path.basename(file_url))
            print(f"Downloading {file_url}...")

            if os.path.exists(filename):
                print(f'Already exists: {filename}')
                continue
            else:
                r = requests.get(file_url, stream=True)
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                continue

    print("Download complete!")

if __name__ == "__main__":
    main()
