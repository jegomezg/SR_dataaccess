import os 
import re
import pytz
import xarray as xr
from datetime import timedelta


from .gliders import utils
from .gliders import filter
from .gliders import updraft


def get_filtered_heights(cbh, cth, clouds_filtered):
    # Ensure CBH and CTH are only considered where clouds_filtered is not null and CBH is below 5000m
    cbh_clouds = cbh.where((cbh.notnull() == clouds_filtered.notnull()) & (cbh <= 5000))
    cth_clouds = cth.where((cth.notnull() == clouds_filtered.notnull()) & (cth <= 5000))
    return cbh_clouds, cth_clouds

def get_height(classification_file: str):
    try:
        cbh = classification_file.cloud_base_height_amsl
        cth = classification_file.cloud_top_height_amsl
    except AttributeError:
        cbh = classification_file.cloud_base_height
        cth = classification_file.cloud_top_height
    return cbh,cth

def find_file_by_date(directory, date, pattern=r'(\d{8})'):
    # Convert the date to a string if it's a datetime object
    if not isinstance(date, str):
        date = date.strftime('%Y%m%d')
    
    # Compile the regular expression pattern for performance
    compiled_pattern = re.compile(pattern)

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Search for the date pattern in the filename
        match = compiled_pattern.search(filename)
        if match and date == match.group(1):
            # If a match is found and the date matches, return the full path
            return os.path.join(directory, filename)

    # If no file is found, return None or raise an error
    return None


def get_mean_height(classification_folder, dates):
    
    result = []
    for date in dates:
        
        classification_path = find_file_by_date(classification_folder, date)
        if classification_path is not None:

            classification = xr.open_dataset(classification_path, engine='netcdf4')
            
            cbh, cth = get_height(classification)

            classes = classification.target_classification

            clouds = classes.where(classes == 1) 

            cbh_clouds, cth_clouds = get_filtered_heights(cbh, cth, clouds)

            # Define your time range for filtering
            start_time = date - timedelta(minutes=60)
            end_time = date + timedelta(minutes=60)
            
            # Use .sel() method to select data within the time range
            cth_filtered_by_time = cth_clouds.sel(time=slice(start_time.replace(tzinfo=None), end_time.replace(tzinfo=None)))

            # Compute the mean cloud top height across all heights for each time point
            cth_mean_time = cth_filtered_by_time.mean(dim='height', skipna=True)

            # Compute the overall mean cloud top height
            overall_mean_cth = cth_mean_time.mean().item()

            # Append the result as a dictionary to the result list
            result.append({'date': date, 'mean_cth': overall_mean_cth})

    return result