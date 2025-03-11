import rasterio
import xarray as xr
import rioxarray as rxr
import numpy as np
from datetime import timezone, datetime
from rasterio.transform import from_bounds

import pvlib
from typing import List, Dict, Tuple


import matplotlib.pyplot as plt


def load_xarray_from_url(image_url: str) -> xr.DataArray:
    gdal_config = {
        "GDAL_HTTP_COOKIEFILE": "~/cookies.txt",
        "GDAL_HTTP_COOKIEJAR": "~/cookies.txt",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
        "GDAL_HTTP_UNSAFESSL": "YES",
    }

    chunk_size = dict(
        band=1, x=512, y=512
    )  # Tiles have 1 band and are divided into 512x512 pixel chunks
    max_retries = 50

    for _i in range(max_retries):
        try:
            with rasterio.Env(**gdal_config):
                image_xarray = rxr.open_rasterio(
                    image_url, chunks=chunk_size, masked=True
                ).squeeze("band", drop=True)
                print(image_xarray.dtype)
                break  # Exit the retry loop if successful
        except Exception as ex:
            print(f"vsi curl error: {ex}. Retrying...")

    return image_xarray


def crop_rioxarray_to_aoi(rxr_data, aoi_gdf):
    """
    Crops a rioxarray to the projected area of interest (AOI) defined by a GeoDataFrame.

    Args:
        rxr_data (rioxarray.DataArray): The rioxarray data to be cropped.
        aoi_gdf (geopandas.GeoDataFrame): The GeoDataFrame representing the AOI.

    Returns:
        rioxarray.DataArray: The cropped rioxarray data.
    """
    # Get the projection of the rioxarray
    rxr_crs = rxr_data.rio.crs
    # Reproject the GeoDataFrame to match the rioxarray projection
    aoi_gdf_projected = aoi_gdf.to_crs(rxr_crs)
    # Get the bounding box of the projected AOI
    aoi_bbox = aoi_gdf_projected.total_bounds
    # Crop the rioxarray to the projected AOI bounding box
    cropped_rxr_data = rxr_data.rio.clip_box(*aoi_bbox)
    return cropped_rxr_data


def extract_bits(data, bit):
    return (data >> bit) & bit


def match_flag_bit(requ_flag):
    match requ_flag:
        case "cirrus":
            return 0
        case "cloud":
            return 1
        case "adjacent":
            return 2
        case "shadow":
            return 3
        case "snow":
            return 4
        case "water":
            return 5


def get_bitmast_from_fmask(fmask, req_flag="cloud"):
    bit = match_flag_bit(req_flag)
    try:
        mask = extract_bits(fmask, bit) << bit
    except TypeError:
        data = fmask.data
        if data.dtype.kind not in 'iu':  # i is for integer, u for unsigned integer
            data = data.astype(int)  # Cast to int to ensure bitwise operations work

        mask = extract_bits(data, bit) << bit
    return xr.where(mask > 0, 1, 0)


##############################################################
# TODO: maybe move this

import json
from datetime import datetime


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        return super().default(obj)


def save_pyrano_polar_to_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, cls=CustomEncoder, indent=2)


def custom_decoder(obj):
    if "__type__" in obj and obj["__type__"] == "datetime":
        return datetime.fromisoformat(obj["value"])
    return obj


def load_pyrano_polar_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file, object_hook=custom_decoder)
    return data


##############################################################
# TODO: maybe move this


def filter_high_variability_high_mean_dates(  # TODO Replace with a more seriuos sky calssification
    y_variable: List[Dict[str, List[float]]],
    variability_threshold: float = 30,
    mean_threshold: float = 0.6,
) -> List[str]:
    """
    Filters dates from y_variable where both 'high variability' and 'high mean' criteria are met.

    Args:
        y_variable (List[Dict[str, List[float]]]): List of dictionaries containing 'date' and 'Kt' keys.
        variability_threshold (float): Threshold for 'high variability' in percentage.
        mean_threshold (float): Threshold for 'high mean'.

    Returns:
        List[str]: Filtered dates where both 'high variability' and 'high mean' criteria are met.
    """
    std_mean: List[Tuple[str, float, float]] = [
        (ye["date"], np.std(ye["Kt"]), np.mean(ye["Kt"])) for ye in y_variable
    ]

    high_variability_high_mean_dates: List[str] = [
        date
        for date, std, mean in std_mean
        if std / mean * 100 > variability_threshold and mean > mean_threshold
    ]

    return high_variability_high_mean_dates


def calculate_solar_angles(date_list, latitude, longitude):
    # Empty list to store the result
    solar_angles_list = []

    # Loop over each datetime in the provided list
    for date in date_list:
        # Convert timezone to UTC
        date_utc = date.astimezone(timezone.utc)

        # Calculate the solar position
        solar_position = pvlib.solarposition.get_solarposition(
            date_utc, latitude, longitude
        )

        # Extract the azimuth and zenith angles from the solar_position DataFrame
        azimuth = solar_position["azimuth"].iloc[0]
        zenith = solar_position["apparent_zenith"].iloc[0]

        # Append the result as a dictionary to the solar_angles_list
        solar_angles_list.append(
            {"date": date_utc, "azimuth": azimuth, "zenith": zenith}
        )

    return solar_angles_list




################ jpeg coresitration ################

import requests
import rioxarray
import json
import xarray as xr
import io
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def get_json_metadata(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()



def georeference_jpg(jpeg_url, metadata_path,target_crs = None):
    jpeg_bytes = download_image(jpeg_url)
    metadata = get_json_metadata(metadata_path)

    with MemoryFile(io.BytesIO(jpeg_bytes)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read().astype('float32')

    target_crs = target_crs if target_crs is not None else CRS.from_epsg(metadata['properties']['proj:epsg'])
    coordinates = metadata['geometry']['coordinates'][0][0]
    west = min(coord[0] for coord in coordinates)  # WEST
    north = max(coord[1] for coord in coordinates)  # NORTH
    east = max(coord[0] for coord in coordinates)  # EAST
    south = min(coord[1] for coord in coordinates)  # SUTH
    transform = from_bounds(west=west, south=south, east=east, north=north, width=data.shape[2], height=data.shape[1])

    xarr = xr.DataArray(
        data,
        dims=('band', 'y', 'x'),
        coords={
            "band": ['R', 'G', 'B'],
            "y": np.linspace(start=north, stop=south, num=data.shape[1]),
            "x": np.linspace(start=west, stop=east, num=data.shape[2])
        }
    )

    xarr = xarr.sortby('y')
    xarr.rio.write_crs(CRS.from_epsg(4326), inplace=True)
    xarr.rio.write_transform(transform, inplace=True)

    test = xr.where(xarr != 0, xarr, np.nan)
    xarr_interpolated = test.interpolate_na(dim="y", method='linear')
    xarr_interpolated.rio.write_crs(CRS.from_epsg(4326), inplace=True)
    xarr_interpolated.rio.write_nodata(256, inplace=True)  # Ensure dtype supports this value

    xarr_reproj = xarr_interpolated.rio.reproject(target_crs)
    nodata_val = xarr_interpolated.rio.nodata
    xarr_reproj.rio.write_nodata(nodata_val, inplace=True)


    xarr_normalized = (xarr_reproj - xarr_reproj.min()) / (xarr_reproj.max() - xarr_reproj.min())
    xarr_normalized.rio.write_nodata(1, inplace=True)
    return xarr_normalized


import pandas as pd
import numpy as np
import numpy as np
import rasterio

import rioxarray
import matplotlib.pyplot as plt



import numpy as np
import xarray as xr
from timezonefinder import TimezoneFinder 
from datetime import timezone,datetime
import pytz
from scipy.ndimage import zoom

from tqdm import tqdm
import pvlib




def calculate_solar_angles(date_list, latitude, longitude):
    # Empty list to store the result
    solar_angles_list = []
    
    # Loop over each datetime in the provided list
    for date in date_list:
        # Convert timezone to UTC
        date_utc = date.astimezone(timezone.utc)
        
        # Calculate the solar position
        solar_position = pvlib.solarposition.get_solarposition(
            date_utc, latitude, longitude
        )
        
        # Extract the azimuth and zenith angles from the solar_position DataFrame
        azimuth = solar_position['azimuth'].iloc[0]
        zenith = solar_position['apparent_zenith'].iloc[0]
        
        # Append the result as a dictionary to the solar_angles_list
        solar_angles_list.append({'date': date_utc, 'azimuth': azimuth, 'zenith': zenith})
    
    return solar_angles_list

def create_aoi_feature(center_lat, center_lon, distance_km):
    """
    Create a GeoJSON-like dictionary for an Area of Interest (AOI) square polygon.

    Parameters:
    - center_lat: Latitude of the center point.
    - center_lon: Longitude of the center point.
    - distance_km: Distance in kilometers from the center to each side of the square.

    Returns:
    - A GeoJSON-like dictionary representing the AOI.
    """
    # Earth's radius in km and Pi
    R = 6378.1
    pi = np.pi

    # Calculate the latitude and longitude offsets for the given distance
    lat_offset = distance_km / 111
    lon_offset = distance_km / (111 * np.cos(center_lat * (pi / 180)))

    # Create the coordinates for the square polygon
    coordinates = [
        [
            [center_lon - lon_offset, center_lat - lat_offset],
            [center_lon - lon_offset, center_lat + lat_offset],
            [center_lon + lon_offset, center_lat + lat_offset],
            [center_lon + lon_offset, center_lat - lat_offset],
            [center_lon - lon_offset, center_lat - lat_offset]  # Close the loop
        ]
    ]

    # Construct the GeoJSON-like dictionary
    aoi_feature = {
        "type": "Polygon",
        "coordinates": coordinates
    }

    return aoi_feature

"""
import boto3
import xml.etree.ElementTree as ET
import mgrs
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta, date,timezone

import numpy as np



def latlon_to_mgrs(lat, lon):
    m = mgrs.MGRS()
    coordinate = m.toMGRS(lat, lon, MGRSPrecision=4)
    return coordinate[:5]

def get_sensed_time_from_metadata(year, month, day, tile):
    s3_client = boto3.client('s3')
    bucket_name = 'sentinel-s2-l2a'
    key = f'tiles/{tile}/{year}/{month}/{day}/0/metadata.xml'
    


    s3_object = s3_client.get_object(Bucket=bucket_name, Key=key, RequestPayer='requester')
    metadata_xml = s3_object['Body'].read().decode('utf-8')

    root = ET.fromstring(metadata_xml)
    namespace = {'n1': 'https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd'}
    sensed_time_element = root.find(".//n1:General_Info/SENSING_TIME", namespace)
    
    if sensed_time_element is not None:
        return sensed_time_element.text.split("T")[1][:8]  # Extract HH:MM
    else:
        return "N/A"

def fetch_s2_days(satellite_dates, mgrs_tile):
    zone = mgrs_tile[:2]
    lat_band = mgrs_tile[2]
    square_id = mgrs_tile[3:]
    data = []

    for date in satellite_dates:
        time = get_sensed_time_from_metadata(date.year, date.month, date.day, f"{zone}/{lat_band}/{square_id}")
        hours, minutes,seconds = time.split(":")
        dt = datetime(int(date.year), int(date.month), int(date.day), int(hours), int(minutes), int(seconds))
        data.append(dt)

    return data"""


def localize_satellite_dates(satellite_list,camera_lat,camera_lon):
    
    
    
    finder  = TimezoneFinder()
    camera_timezone = finder.timezone_at(lng=camera_lon, lat=camera_lat)
    camera_timezone_tz = pytz.timezone(camera_timezone)
    satellite_dates = [date.replace(tzinfo=timezone.utc) for date in satellite_list]
    return [date.astimezone(camera_timezone_tz) for date in satellite_dates]


def metadata_sensing_times(satellite_list, lat,lon):
    
    satellite_list =[
            datetime.utcfromtimestamp(dt.astype("O") / 1e9).replace(tzinfo=timezone.utc)
            for dt in satellite_list
        ]
    
    # TODO not pass complete years only 
    mgrs_tile = latlon_to_mgrs(lat,lon)
    corrected_dates = fetch_s2_days(satellite_list,mgrs_tile)

    corrected_dates = np.array(
        [
            dt.replace(tzinfo=timezone.utc)
            for dt in corrected_dates
        ]
    )
    
    min_date = satellite_list[0]
    max_date = satellite_list[-1]

    # Filter the dates to only include those within the time range of satellite_list
    filtered_dates = [date for date in corrected_dates if min_date <= date <= max_date]

    located_dates = localize_satellite_dates(filtered_dates,lat,lon)

    return located_dates




def ghi_dem_at_location(dem_reprojected,timestamp):    
    nodata = dem_reprojected.rio.nodata
    transform = dem_reprojected.rio.transform()

    # Generate latitude and longitude arrays from the elevation DataArray
    if len(dem_reprojected.shape) == 3:
        dem_reprojected=dem_reprojected.squeeze()
        
    rows, cols = np.indices(dem_reprojected.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lats, lons = np.array(ys), np.array(xs)

    # Filter out no data values from the elevation
    mask = dem_reprojected.data != nodata  # Ensure that you're using the underlying numpy array
    valid_elevation = dem_reprojected.data[mask]  # Index with the mask to get valid elevations as a 1D array
    valid_lats = lats[mask]
    valid_lons = lons[mask]

    # Sample random points for Linke turbidity calculation (e.g., 100 points)
    num_samples = 100
    random_indices = np.random.choice(len(valid_lats), size=num_samples, replace=False)
    sampled_lats = valid_lats[random_indices]
    sampled_lons = valid_lons[random_indices]

    # Prepare the solar position calculation for sampled points
    date = pd.Timestamp(timestamp)
    times = pd.DatetimeIndex([date] * num_samples)

    # Calculate Linke turbidity for sampled points using tqdm for progress tracking
    linke_turbidities = []
    for idx in tqdm(range(num_samples), desc='Calculating Linke turbidity'):
        time = pd.DatetimeIndex([date])
        lat = sampled_lats[idx]
        lon = sampled_lons[idx]
        lt = pvlib.clearsky.lookup_linke_turbidity(time, lat, lon).values[0]
        linke_turbidities.append(lt)

    # Convert list to numpy array
    linke_turbidities = np.array(linke_turbidities)

    # Take the average Linke turbidity
    average_linke_turbidity = np.mean(linke_turbidities)

    # Take the average Linke turbidity
    average_linke_turbidity = np.mean(linke_turbidities)

    # Prepare the solar position calculation for all points
    times_full = pd.DatetimeIndex([date] * len(valid_lats))
    solar_position = pvlib.solarposition.get_solarposition(
        time=times_full,
        latitude=valid_lats,
        longitude=valid_lons,
        altitude=valid_elevation,
        temperature=20
    )

    # Calculate the relative airmass
    relative_airmass = pvlib.atmosphere.get_relative_airmass(solar_position['apparent_zenith'])
    # Calculate the pressure in Pascals at the given elevation
    pressure = pvlib.atmosphere.alt2pres(valid_elevation)
    # Calculate the absolute airmass
    absolute_airmass = pvlib.atmosphere.get_absolute_airmass(relative_airmass, pressure)
    # Calculate extra terrestrial radiation
    dni_extra = pvlib.irradiance.get_extra_radiation(date)

    # Calculate the clear sky GHI using the Ineichen model with average Linke turbidity
    clearsky_ghi = pvlib.clearsky.ineichen(
        solar_position['apparent_zenith'],
        absolute_airmass,
        linke_turbidity=average_linke_turbidity,
        altitude=valid_elevation,
        dni_extra=dni_extra
    )

    # Now, map the calculated clearsky GHI values back to the original grid
    solar_radiation = xr.full_like(dem_reprojected, nodata, dtype=np.float32)
    solar_radiation.values[mask] = clearsky_ghi['ghi']

    # Add the original geographical information back to the new DataArray
    solar_radiation.attrs.update(dem_reprojected.attrs)

    # Set the correct CRS for solar_radiation if it's known (e.g., EPSG:32631)
    solar_radiation.rio.write_crs("EPSG:4326", inplace=True)
    
    return solar_radiation

import rioxarray
from rasterio.enums import Resampling

def reproject_dem_to_cob_wgs84(tiff_file,cob):
    dem = rioxarray.open_rasterio(tiff_file)

    # Squeeze the single-band dimension if present
    dem_data_2d = dem.squeeze()

    # Reproject the DEM to WGS84
    dem_reprojected = dem_data_2d.rio.reproject("EPSG:4326", resampling=Resampling.bilinear)

    scl_band = cob.sel(band='SCL').isel(time=0)
    scl_band = scl_band.drop_vars([coord for coord in scl_band.coords if coord not in ['x', 'y']])
    scl_band.rio.write_crs("EPSG:32631", inplace=True)

    # If 'cob' is an xarray.DataArray with a defined CRS, reproject it to WGS84
    cob_reprojected = scl_band.rio.reproject("EPSG:4326")

    # Now to clip the DEM to the bounds of the COB, we need the bounds in WGS84
    x_min, x_max = cob_reprojected.x.min().item(), cob_reprojected.x.max().item()
    y_min, y_max = cob_reprojected.y.min().item(), cob_reprojected.y.max().item()
    cob_bounds_wgs84 = (x_min, y_min, x_max, y_max)

    # Clip the reprojected DEM to the bounds of the reprojected COB
    dem_reprojected_clipped = dem_reprojected.rio.clip_box(*cob_bounds_wgs84)
    
    return dem_reprojected_clipped


import numpy as np
import xarray as xr

def shift_image_by_solar_position(image, solar_azimuth, solar_zenith, cloud_height, resolution=10):
    """
    Shifts an xarray.DataArray image based on solar azimuth and solar zenith angles.
    
    Parameters:
    image (xarray.DataArray): The input image with spatial coordinates.
    solar_azimuth (float): Solar azimuth angle in degrees.
    solar_zenith (float): Solar zenith angle in degrees.
    cloud_height (float): Average cloud height in meters.
    resolution (float): Resolution of the image in meters per pixel.
    
    Returns:
    xarray.DataArray: The shifted image.
    """
    
    # Convert angles from degrees to radians
    solar_azimuth_rad = np.deg2rad(solar_azimuth)
    solar_zenith_rad = np.deg2rad(90 - solar_zenith)  # Convert zenith to elevation angle
    shadow_length = cloud_height / np.tan(solar_zenith_rad)
    
    
    # Calculate the shadow offsets in meters
    x_shift_meters = shadow_length * -np.sin(solar_azimuth_rad)
    y_shift_meters = shadow_length * np.cos(solar_azimuth_rad)
    
    # Calculate the shift in coordinate units
    x_shift_units = x_shift_meters / resolution
    y_shift_units = y_shift_meters / resolution
    
    # Create new x and y coordinates
    new_x = image.x + x_shift_units
    new_y = image.y + y_shift_units
    
    # Assign the new coordinates to the image
    shifted_image = image.assign_coords({'x': new_x, 'y': new_y})
    
    return shifted_image