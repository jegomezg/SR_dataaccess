import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import ocf_blosc2
import pyproj
import pandas as pd
import numpy as np
import datetime

GEOSTATIONARY_1KM_PROJ_SPEC = {
    "description": "MSG SEVIRI Rapid Scanning Service area definition with 1 km resolution",
    "projection": {
        "proj": "geos",
        "lon_0": 9.5,
        "h": 35785831,
        "x_0": 0,
        "y_0": 0,
        "a": 6378169,
        "rf": 295.488065897014,
        "no_defs": True,
        "type": "crs",
    },
    "shape": {"height": 4176, "width": 5568},
    "area_extent": {
        "lower_left_xy": [2862884.573638439, 5571248.390376568],
        "upper_right_xy": [-2705863.4808659554, 1394687.349498272],
        "units": "m",
    },
}

GEOSTATIONARY_3KM_PROJ_SPEC = {
    "description": "MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution",
    "projection": {
        "proj": "geos",
        "lon_0": 9.5,
        "h": 35785831,
        "x_0": 0,
        "y_0": 0,
        "a": 6378169,
        "rf": 295.488065897014,
        "no_defs": None,
        "type": "crs",
    },
    "shape": {"height": 1392, "width": 3712},
    "area_extent": {
        "lower_left_xy": [5567248.074173927, 5570248.477339745],
        "upper_right_xy": [-5570248.477339745, 1393687.2705221176],
        "units": "m",
    },
}

# Function to return the correct URL and projection based on dataset type
def get_dataset_link_and_projection(dataset_type, date):
    if dataset_type.lower() == 'hrv':
        METEOSAT_HRV_BASE_URL = f"gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{date.year}_hrv.zarr"
        projection = GEOSTATIONARY_1KM_PROJ_SPEC
        return METEOSAT_HRV_BASE_URL, projection
    else:
        METEOSAT_NO_HRV_BASE_URL = f"gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{date.year}_nonhrv.zarr"
        projection = GEOSTATIONARY_3KM_PROJ_SPEC
        return METEOSAT_NO_HRV_BASE_URL, projection

def transform_bbox_to_geostationary(bbox, src_crs, geostationary_proj):
    # Create a transformer to transform coordinates from the source CRS to the geostationary projection
    transformer = pyproj.Transformer.from_crs(
        src_crs,
        pyproj.CRS.from_dict(geostationary_proj['projection']),
        always_xy=True
    )
    # Transform the bounding box coordinates
    lower_left_xy = transformer.transform(bbox[0], bbox[1])
    upper_right_xy = transformer.transform(bbox[2], bbox[3])
    return lower_left_xy, upper_right_xy

def subset_data_with_coords(data_array, x_min, x_max, y_min, y_max):
    subset = (data_array
        .sel(
            x_geostationary=slice(int(x_max), int(x_min)),
            y_geostationary=slice(int(y_min), int(y_max))
        )
    )
    return subset

# Function to select the closest time frame without timezone issues
def select_closest_time(data_array, target_time):
    # Convert target_time to numpy datetime64
    target_np_time = np.datetime64(target_time)
    
    # Get the available times in the dataset
    dataset_times = data_array.time.values
    
    # Find the index of the closest time
    # First convert everything to integers (nanoseconds since epoch)
    target_int = target_np_time.astype('datetime64[ns]').astype(np.int64)
    times_int = np.array([np.datetime64(t).astype('datetime64[ns]').astype(np.int64) for t in dataset_times])
    
    # Find the closest match
    closest_idx = np.argmin(np.abs(times_int - target_int))
    closest_time = dataset_times[closest_idx]
    
    # Return the data at the closest time
    return data_array.sel(time=closest_time)

def subset_and_select_closest_data(data_array, bbox, src_crs, geostationary_proj, target_time):
    # Transform the bounding box coordinates to the geostationary projection
    lower_left_xy, upper_right_xy = transform_bbox_to_geostationary(bbox, src_crs, geostationary_proj)
    x_min, y_min = lower_left_xy
    x_max, y_max = upper_right_xy
    
    # Subset the data spatially
    spatial_subset = subset_data_with_coords(data_array, x_min, x_max, y_min, y_max)
    
    # Select the closest time frame
    closest_data = select_closest_time(spatial_subset, target_time)
    
    return closest_data

def get_subset_meteosat(date, bbox, src_crs, data_type):
    # Ensure date is handled correctly - we'll use it directly as is
    # since it's already a datetime.datetime with tzinfo=UTC
    
    base_url, geoproj = get_dataset_link_and_projection(data_type, date)
    
    dataset = xr.open_dataset(
        base_url, 
        engine="zarr", 
        chunks="auto",
    )
    data_array = dataset['data']
    
    return subset_and_select_closest_data(data_array, bbox, src_crs, geoproj, date)