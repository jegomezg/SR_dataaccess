import os
import numpy as np
import pandas as pd
import xarray as xr
import pvlib  # Assuming pvlib is used for getting CAMS data
from datetime import datetime


def download_solar_data(
    lat, lon, start_date, end_date, bbox, gsd, email, output_folder
):
    # Convert start and end dates from strings to datetime objects if they are not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Define the bounding box and grid size
    bbox = 32000
    gsd = 2000

    # Calculate number of grid points
    num_lat_points = int(bbox / gsd) + 1
    num_lon_points = int(bbox / gsd) + 1

    # Calculate conversion factors
    lat_conversion = gsd / 111320
    lon_conversion = gsd / (40075000 * np.cos(np.radians(lat)) / 360)

    # Create grid points
    lat_points = np.linspace(
        lat - (lat_conversion * (num_lat_points - 1) / 2),
        lat + (lat_conversion * (num_lat_points - 1) / 2),
        num_lat_points,
    )
    lon_points = np.linspace(
        lon - (lon_conversion * (num_lon_points - 1) / 2),
        lon + (lon_conversion * (num_lon_points - 1) / 2),
        num_lon_points,
    )

    # Check if the output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fetch the data and populate the grid
    for lat in lat_points:
        for lon in lon_points:
            filename = os.path.join(output_folder, f"solar_data_{lat}_{lon}.nc")
            if not os.path.exists(filename):
                try:
                    # Retrieve data for the current grid point (assuming pvlib.iotools.get_cams is the correct method)
                    cams_map_data, cams_metadata = pvlib.iotools.get_cams(
                        latitude=lat,
                        longitude=lon,
                        start=start_date,
                        end=end_date,
                        email=email,
                        identifier="cams_radiation",
                        time_step="15min",
                        time_ref="UT",
                        server="api.soda-pro.com",
                    )

                    # Additional data processing here...

                    # Save this dataset as a NetCDF file
                    ds.to_netcdf(filename)
                    print(f"Saved {filename}")

                except Exception as e:
                    print(f"Error retrieving data for point ({lat}, {lon}): {e}")
            else:
                print(f"File {filename} already exists")


def preprocess(ds):
    return ds.chunk({"time": "auto"})


def match_allsky_with_cams(allsky_dataarray, cams_folder_path):
    """
    Match an allsky xarray DataArray with a CAMS xarray dataset for the first `num_matches` times.
    The CAMS dataset is opened, matched, and then closed after matching.

    Parameters:
    allsky_dataarray (xarray.DataArray): The allsky data with a time dimension to match.
    cams_folder_path (str): The folder path where CAMS NetCDF files are stored.
    num_matches (int): The number of times to match.

    Returns:
    list of tuples: A list of tuples with matched time, allsky data, and interpolated CAMS data.
    """
    match_time = allsky_dataarray.time.values

    with xr.open_mfdataset(
        os.path.join(cams_folder_path, "solar_data_*.nc"),
        combine="by_coords",
        preprocess=preprocess,
        parallel=True
    ) as cams_dataset:
        cams_dataset["time"] = pd.to_datetime(cams_dataset["time"].values)

    # Interpolate CAMS data to the allsky time
    return cams_dataset.ghi.interp(time=pd.to_datetime(match_time)).isel(altitude=0)
