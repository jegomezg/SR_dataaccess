from cloudnetpy.products import generate_classification
from cloudnetpy.plotting import generate_figure
from IPython.display import IFrame
from unidecode import unidecode
from typing import Optional
import xarray as xr
import pandas as pd
import requests
import fnmatch
import wget
import re
import os


def is_cloudnetpy_dir(site:str, categorize_dir:str='Categorize', classification_dir:str='Classification'):    
    
    """Check if the classification files in a directory are cloudnetpy format.
    If not, generates Cloudnet Level 1b netCDF files from categorize files.
    Needed for legacy data (e.g. Hyytiala 2014-2017) as well as ARM data and other sources. 
    """
    
    for fileclass in os.listdir(classification_dir):
        classification_xrfile = xr.open_dataset(classification_dir + '/' + fileclass)
        if len(classification_xrfile.dims) == 0:
            pass
        else:
            classification_cloudnetpy_exist = any('cloudnetpy' in i for i in classification_xrfile.attrs)
            if not classification_cloudnetpy_exist:
                date = fileclass[0:8]
                classification_xrfile.close()
                os.remove(os.path.join(classification_dir,fileclass))
                for filecat in os.listdir(categorize_dir):
                    if filecat.startswith(date):
                        generate_classification(os.path.join(categorize_dir, filecat), os.path.join(classification_dir, filecat[:8] + '_' + site + '_classification' + '.nc'))

def is_cloudnetpy_file(classification_file: str, categorize_file: str):
    output_name = str(classification_file)
    classification_xrfile = xr.open_dataset(classification_file)
    if len(classification_xrfile.dims) == 0:
        os.remove(classification_file)
        os.remove(categorize_file)
    else:
        classification_cloudnetpy_exist = any('cloudnetpy' in i for i in classification_xrfile.attrs)
        if not classification_cloudnetpy_exist:
            classification_xrfile.close()
            os.remove(classification_file)
            generate_classification(categorize_file, output_name)

def open_files(classification_file: str, categorize_file: str, start_hour: int = None, end_hour: int = None):
    is_cloudnetpy_file(classification_file, categorize_file)
    
    classification = xr.open_dataset(classification_file)
    categorize = xr.open_dataset(categorize_file)
    
    # Extract the date from the 'time' variable
    split = str(categorize.time.values[0])[:10].split('-')
    date = ''.join(split)
    
    # Extract the site information from the classification filename
    output_name = str(classification_file)
    site = output_name.split('_')[-2]
    
    # If start_hour and end_hour are specified, filter the datasets by this range
    if start_hour is not None and end_hour is not None:
        # Convert start and end hours to time strings on the date of the dataset
        start_time_str = f"{date}T{start_hour:02d}:00:00"
        end_time_str = f"{date}T{end_hour:02d}:00:00"
        
        # Convert the time strings to datetime objects
        start_time = xr.coding.times.parse_iso8601(start_time_str)
        end_time = xr.coding.times.parse_iso8601(end_time_str)
        
        # Select the data between the start and end times
        classification = classification.sel(time=slice(start_time, end_time))
        categorize = categorize.sel(time=slice(start_time, end_time))
    
    return classification, categorize, date, site

# Usage example:
# classification_subset, categorize_subset, date, site = open_files(
#     'classification_file.nc', 'categorize_file.nc', start_hour=12, end_hour=14
# )


def get_height(classification_file: str):
    cbh = classification_file.cloud_base_height_amsl
    cth = classification_file.cloud_top_height_amsl
    return cbh,cth

def get_doppler(categorize):
    doppler_vel = categorize.v
    return doppler_vel
    

def get_classes(classification_file: str):
    classes = classification_file.target_classification
    
    #Subset specific classes
    clouds = classes.where(classes == 1) 
    aerosols = classes.where(classes == 8)
    insects = classes.where((classes == 9) | (classes == 10))
    drizzle = classes.where((classes == 2) | (classes == 3))
    ice = classes.where((classes == 4) | (classes == 5) | (classes == 6) | (classes == 7))
    fog = clouds.where(clouds.height < 400)

    return classes, clouds, aerosols, insects, drizzle, ice, fog


def preprocess(classification_file: str, categorize_file: str):
    is_cloudnetpy_file(classification_file, categorize_file)
    classification = xr.open_dataset(classification_file)
    categorize = xr.open_dataset(categorize_file)
    classes = classification.target_classification
    clouds = classes.where(classes == 1) 
    aerosols = classes.where(classes == 8)
    insects = classes.where((classes == 9) | (classes == 10))
    drizzle = classes.where((classes == 2) | (classes == 3))
    ice = classes.where((classes == 4) | (classes == 5) | (classes == 6) | (classes == 7))
    fog = clouds.where(clouds.height < 400)
    cbh = classification.cloud_base_height_amsl
    return classification, categorize, clouds, aerosols, insects, drizzle, ice, fog, cbh

def download_arm(lidar: None, doppler: None, skycam: None):
    """Download Atmospheric Radiation Measurement (ARM) data
    """
import os
import requests
import pandas as pd
import fnmatch
import wget
from typing import Optional

import os
import requests
import pandas as pd
import wget
from datetime import datetime, timedelta
from typing import Optional
import fnmatch

def download_cloudnet(
    site: str,
    start: str,
    end: Optional[str] = None,
    output_dir: str = '',
    products: list = ['categorize', 'classification'],
):
    """Query data from Cloudnet data portal API and download files if they don't exist.

    Args:
        site (str): Field site
        start (str): Start date in yyyy-mm-dd (e.g. 2014-02-02)
        end (str): End date in yyyy-mm-dd (for a range of dates if requested)
        products [list]: default products include categorize and classification files

    Example: download_cloudnet(site='hyytiala', start='2014-08-24', end='2014-08-31')

    Reference: https://docs.cloudnet.fmi.fi/api/data-portal.html
    """

    # Function to generate all dates in the range
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)

    # Convert start date string to datetime object
    start_date = datetime.strptime(start, '%Y-%m-%d')
    # If end date is not provided, use start date as end date
    end_date = datetime.strptime(end, '%Y-%m-%d') if end else start_date


    os.makedirs(output_dir, exist_ok=True)

    existing_files = False

    # Generate list of expected filenames for the date range
    for single_date in daterange(start_date, end_date):
        for product in products:
            expected_filename = f'{site}_{single_date.strftime("%Y%m%d")}_{product}.nc'
            if os.path.exists(os.path.join(output_dir, product.capitalize(), expected_filename)):
                existing_files = True

    # If files for the date range already exist, skip querying the API
    if existing_files:
        return

    # Perform API query and download if necessary
    legacy_years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
    url = f'https://cloudnet.fmi.fi/api/files?site={site}'
    payload = {
        'date': start,
        'product': products
    } if end is None else {
        'dateFrom': start,
        'dateTo': end,
        'product': products
    }
    if any(year in start for year in legacy_years):
        url += '&showLegacy'

    response = requests.get(url, params=payload)
    data = response.json()
    df = pd.DataFrame(data)

    if not df.empty:
        for file_url in df['downloadUrl']:
            product_type = 'Categorize' if 'categorize' in file_url else 'Classification'
            product_dir = os.path.join(output_dir, product_type)
            os.makedirs(product_dir, exist_ok=True)

            filename = os.path.join(product_dir, os.path.basename(file_url))
            if not os.path.exists(filename):
                wget.download(file_url, filename)
            else:
                pass                
            
def list_sites(type:str):
    """
    Types are: 
    arm: Atmospheric Radiation Measurement (ARM) site.
    campaign: Temporary measurement sites.
    cloudnet: Official Cloudnet sites.
    hidden: Sites that are not visible in the Cloudnet data portal GUI.
    
    Ref: https://docs.cloudnet.fmi.fi/api/data-portal.html#get-apisites--site
    """
    url = 'https://cloudnet.fmi.fi/api/sites'
    payload = {'type': type}
    response = requests.get(url,payload)
    data = response.json()
    df = pd.DataFrame(data)
    return df

def above_ground(file:str):
    """
    Returns a file with a height above ground level 
    """
    ds = xr.open_dataset(file)
    file_above_ground = ds.assign_coords(height = ds.height - ds.altitude)
    return file_above_ground

def download_cloudnet_all(start:str, end:str):
    """
    Example: download_cloudnet_all(start='2022-08-15', end='2022-08-30')
    """
    df = list_sites('cloudnet')
    sites = list(df.id)
    for site in sites:
        download_cloudnet(site=site, start=start, end=end)
        

def purge(dir, pattern):
    for f in os.listdir(dir):
        for i in pattern:
            if re.search(i, f):
                os.remove(os.path.join(dir, f))
