"""
This script downloads the raw 9505 data from the HydroSource2 website.
"""

import requests
from bs4 import BeautifulSoup
import os
from toolkit import outputs_path

### Settings ###
url = "https://hydrosource2.ornl.gov/files/SWA9505V3Flow/" # URL to 9505 endpoint

HUC2S = ["11", "12", "13"] # HUC2 codes for the basins of interest

### Path Configuration ###
download_folder = outputs_path / "9505" / "raw" # Folder to download the data to

### Functions ###
def filter_urls_by_huc2(urls, huc2_codes):
    filtered_urls = []
    for url in urls:

        filename = url.split("/")[-1] 
        huc8_code = filename.split("_")[2]  
        huc2_code = huc8_code[:2]  
        
        if huc2_code in huc2_codes:
            filtered_urls.append(url)

    return filtered_urls


### Main ###

# Get the page contents
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all .nc file links
nc_folders = [url + link.get("href") for link in soup.find_all("a") if link.get("href").endswith("/")]

# Download each .nc file
os.makedirs(download_folder, exist_ok=True)

for folder in nc_folders:
    folder_name = os.path.basename(os.path.normpath(folder))
    os.makedirs(os.path.join(download_folder, folder_name), exist_ok=True)
    response = requests.get(folder)
    soup = BeautifulSoup(response.text, "html.parser")
    nc_files = [folder + link.get("href") for link in soup.find_all("a") if link.get("href").endswith(".nc")]
    nc_files = filter_urls_by_huc2(nc_files, HUC2S)
    
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
