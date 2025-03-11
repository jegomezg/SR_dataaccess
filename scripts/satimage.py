import rioxarray as rxr
import xarray as xr
import rasterio
from rioxarray.merge import merge_arrays
import os
import numpy as np
from datetime import datetime

import geopandas as gpd

from typing import Union, Optional, List
from abc import ABC, abstractmethod
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
import earthaccess
import pandas as pd

import concurrent.futures

from . import utils


SUPPORTED_SATELLITE_MISIONS = {
    "HLS": {
        "long_name": "Harmonized Landas_Sentinel",
        "providers": {"earth_access": "cloud_native_nc"},
        "satellite_type": "POL",
    },
    "MSG": {
        "long_name": "Harmonized Landas_Sentinel",
        "providers": {"earth_access": "cloud_native_nc"},
        "satellite_type": "GEO",
    },
}


class SatelliteImage(ABC):
    """Base class for satellite access. Characterizes supported sensor/mission and parses geographical and temporan infomration

    Args:
        sensor (str): The name of the sensor or mission
        lat (float): central latitud of the AOI
        lon (float): central longitud of ht AOI
        size (int): size of the bbox in meters (diameter)
        start_date (Union[str,datetime]): start date to search fro satellite data
        end_date (Union[str,datetime]): end date to search fro satellite data
        provider (str): if specified and data has multiple provders, whici provider to retrive the data from

    """

    def __init__(
        self,
        sensor: str,
        lat: float,
        lon: float,
        size: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        provider: Optional[str] = None,
    ):
        self.sensor = sensor
        self.description = self._characterize_sensor(sensor)
        self.bbox = self._create_bbox(lat, lon, size)
        self.time_range = self._parse_timerange(start_date, end_date)

    def _characterize_sensor(self, sensor: str) -> dict:
        if sensor not in SUPPORTED_SATELLITE_MISIONS.keys():
            raise ValueError(
                f"Sensor or mission not supported, must be one of {list(SUPPORTED_SATELLITE_MISIONS.keys())}"
            )
        return SUPPORTED_SATELLITE_MISIONS[sensor]

    def _parse_timerange(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> list[datetime, datetime]:
        parsed_dates = []

        for date in [start_date, end_date]:
            if isinstance(date, str):
                try:
                    parsed_date = datetime.strptime(date, "%Y-%m-%d")
                    parsed_dates.append(parsed_date)
                except ValueError:
                    raise ValueError(
                        f"Invalid date format for '{date}'. Expected format: 'YYYY-MM-DD'"
                    )
            elif isinstance(date, datetime):
                parsed_dates.append(date)
            else:
                raise TypeError(
                    f"Invalid type for date: {type(date)}. Expected str or datetime."
                )

        return parsed_dates

    def _create_bbox(
        self, lat: float, lon: float, edge_size: Union[float, int]
    ) -> gpd.GeoDataFrame:
        """Creates a GeoPandas DataFrame containing a Bounding Box (BBox) given a pair of coordinates and a buffer distance.

        Parameters
        ----------
        lat : float
            Latitude.
        lon : float
            Longitude.
        edge_size : float
            Buffer distance in meters.

        Returns
        -------
        gpd.GeoDataFrame
            GeoPandas DataFrame containing the BBox geometry, UTM coordinates, and EPSG code.
        """
        # Get the UTM EPSG from latlon
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(lon, lat, lon, lat),
        )
        epsg = utm_crs_list[0].code
        utm_crs = CRS.from_epsg(epsg)

        # Initialize a transformer to UTM
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        utm_coords = transformer.transform(lon, lat)

        # Create BBox coordinates according to the edge size
        buffer = edge_size / 2
        E = utm_coords[0] + buffer
        W = utm_coords[0] - buffer
        N = utm_coords[1] + buffer
        S = utm_coords[1] - buffer

        polygon = Polygon(
            [
                (W, S),
                (E, S),
                (E, N),
                (W, N),
                (W, S),
            ]
        )

        bbox_gdf = gpd.GeoDataFrame(
            data={"utm_x": [utm_coords[0]], "utm_y": [utm_coords[1]], "epsg": [epsg]},
            geometry=[polygon],
            crs=utm_crs,
        )

        return bbox_gdf

    @property
    def satellite_type(self):
        return self.description["satellite_type"]


# TODO: The function is harcoded to sork with HLS data but i should be generalized -> create a small wrapper for earth access and add other collections as needed
# TODO: nc reader should also be a separated class and improved for accessing and parallell processing
# TODO: Maybe this should not be a subclass but just part of the main one, using sensor to dinamically create the other methods


class HLSImage(SatelliteImage):
    def __init__(
        self,
        sensor: str,
        lat: float,
        lon: float,
        size: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        provider: Optional[str] = None,
    ):
        super().__init__(sensor, lat, lon, size, start_date, end_date)

        self.satellite_list = self._search_satellite_data()

    def _search_satellite_data(self) -> pd.DataFrame:
        """
        Search for satellite data based on the given latitude, longitude, and date range.

        Parameters
        ----------
        center_lat : float
            Latitude of the center point.
        center_lon : float
            Longitude of the center point.
        start_date : str or datetime
            Start date of the date range (inclusive).
        end_date : str or datetime
            End date of the date range (inclusive).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the normalized satellite data.
        """
        if isinstance(self.time_range[0], datetime):
            start_date = self.time_range[0].strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(self.time_range[1], datetime):
            end_date = self.time_range[1].strftime("%Y-%m-%dT%H:%M:%S")

        # Log in to Earthaccess
        earthaccess.login(strategy="netrc", persist=True)

        # Search for satellite data
        print("Searching for scenes over AOI")
        results = earthaccess.search_data(
            short_name=["HLSL30", "HLSS30"],
            bounding_box=tuple(self.bbox.to_crs("EPSG:4326").total_bounds),
            temporal=(start_date, end_date),
            count=2000,
        )
        sat_pd = pd.json_normalize(results)
        # Convert the 'umm.TemporalExtent.RangeDateTime.BeginningDateTime' column to datetime
        sat_pd["umm.TemporalExtent.RangeDateTime.BeginningDateTime"] = pd.to_datetime(
            sat_pd["umm.TemporalExtent.RangeDateTime.BeginningDateTime"]
        )
        
        sat_pd['datetime_group'] = sat_pd['umm.TemporalExtent.RangeDateTime.BeginningDateTime'].dt.strftime('%Y-%m-%d %H:%M')

        return sat_pd

    def _match_date_sat_pd(self, date: datetime):
        date_time_str = date.strftime('%Y-%m-%d %H:%M')
        return self.satellite_list[
            self.satellite_list["datetime_group"] == date_time_str
        ]

    def _get_single_band_url(self, related_urls, band="Fmask"):
        if band == "jpg":
            jpg_urls = [
                item["URL"]
                for item in related_urls
                if "jpg" in item["URL"] and "s3" not in item["URL"]
            ]
            json_urls = [
                item["URL"]
                for item in related_urls
                if "json" in item["URL"] and "s3" not in item["URL"]
            ]
            return list(zip(jpg_urls, json_urls))
        else:
            fmask_urls = [
                item["URL"]
                for item in related_urls
                if band in item["URL"] and "s3" not in item["URL"]
            ]
            return fmask_urls[0] if fmask_urls else None


    def _load_and_process_image(self, url, crop_aoi, band):
        if band == 'jpg':
            image_xarray = utils.georeference_jpg(url[0][0], url[0][1], self.bbox.crs)
            if crop_aoi is not None:
                image_xarray = utils.crop_rioxarray_to_aoi(image_xarray, crop_aoi)
        else:
            image_xarray = utils.load_xarray_from_url(url)
            if crop_aoi is not None:
                image_xarray = utils.crop_rioxarray_to_aoi(image_xarray, crop_aoi)
        return image_xarray
    
    def load_image_from_date(self, date: datetime, bands: list[str] = ["Fmask"]):
        sat_pd_registers = self._match_date_sat_pd(date)

        def process_band(band):
            urls = (
                sat_pd_registers["umm.RelatedUrls"]
                .apply(lambda x: self._get_single_band_url(x, band=band))
                .tolist()
            )
            num_images = len(urls)

            if num_images == 0:
                raise ValueError(f"No satellite image registers provided for band '{band}'.")

            print(f"{'Single' if num_images == 1 else num_images} scene{'s' if num_images > 1 else ''} for band {band}")

            band_xarrays = [self._load_and_process_image(url, self.bbox, band) for url in urls]

            if num_images > 1:
                if band == "Fmask":
                    merged_xarray = merge_arrays(band_xarrays, nodata=255).astype("uint8")
                else:
                    merged_xarray = merge_arrays(band_xarrays, method='last')
            else:
                merged_xarray = band_xarrays[0]

            if band != 'jpg':
                merged_xarray = merged_xarray.assign_coords(band=band)

            return merged_xarray

        # Process bands sequentially
        image_xarrays = [process_band(band) for band in bands]

        # Align all bands to a common grid
        reference = image_xarrays[0]
        aligned_xarrays = [
            xr.align(reference, arr, join='outer')[1].rio.reproject_match(reference)
            for arr in image_xarrays
        ]

        # Verify shapes and coordinates match
        shapes = [arr.shape for arr in aligned_xarrays]
        if len(set(shapes)) != 1:
            raise ValueError(f"Mismatched shapes after alignment: {shapes}")

        # Concatenate the aligned xarrays
        concatenated_xarray = xr.concat(aligned_xarrays, dim="band")

        return concatenated_xarray