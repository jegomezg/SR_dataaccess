from pathlib import Path
import pandas as pd
import numpy as np
from timezonefinder import TimezoneFinder
from datetime import timezone, datetime
import pytz


from . import pyrano
from . import satimage
from . import utils
from . import cloud

from .gliders import utils as glaiderutils


class ObsSite:
    """An observsation site composed by one or more instruments"""

    def __init__(
        self,
        site_name: str,
        sky_cam=None,
        pyrano: pyrano.Pyranometer = None,
        pyranocam=None,
        cloudnet_name=None,
        satellites: list[satimage.SatelliteImage] = None,
    ) -> None:
        """Initialize the obervation site with one or more instruments

        Args:
            sky_cam (_type_, optional): _description_. Defaults to None.
            pyrano (_type_, optional): _description_. Defaults to None.
            pyranocam (_type_, optional): _description_. Defaults to None.
            satellites (_type_, optional): _description_. Defaults to None.
        """
        self.site_name = site_name
        self.sky_cam = sky_cam
        self.pyrano = pyrano
        self.pyranocam = pyranocam
        self.cloudnet_name = cloudnet_name
        self.satellites = satellites
        self.folder = self._create_output_folder()

    def _create_output_folder(self, output_folder: str = None):
        if output_folder is None:
            # If output_folder is not provided, use the project folder as the default
            project_folder = Path.cwd()  # Get the current working directory
            folder_path = project_folder / "obsite" / self.site_name
        else:
            # If output_folder is provided, use it as the parent folder
            folder_path = Path(output_folder) / "obsite" / self.site_name

        if not folder_path.exists():
            folder_path.mkdir(parents=True)
            print(
                f"Observation site '{self.site_name}' created successfully at {folder_path}."
            )
        else:
            print(
                f"Observation site '{self.site_name}' already exists at {folder_path}."
            )
        return folder_path

    def extract_closest_values(
        self,
        variable: str = "GHI",
        range_in_minutes: int = 0,
        threshold: float = None,
        filter: bool = False,
    ):
        """
        Extract values from an xarray dataset at the nearest available points to a list of dates,
        ensuring that the closest value is always the central value of the returned list within
        the specified range in minutes. If the variable is 'Kc', only values below the threshold are returned.

        Parameters:
        pyrano_nc (xarray.Dataset): The input xarray dataset with a time coordinate.
        sat_dataframe (list of datetime.datetime): Pandas Dataframe with resulting search from nasa earth access.
        variable (str): The variable to extract from the dataset.
        range_in_minutes (int): The range in minutes around the closest value to include in the results.
        threshold (float): The threshold value to filter the variable results. Values above this threshold are ignored.
        filter (bool); rather to filter or not the dates based on pyranometer obervations variability
        Returns:
        list of dicts: A list of dictionaries, each containing the index 'date' and variable values
                        extracted within the specified range, with the closest value at the center.
        """
        PYRANO_SATELLITE_LIST_PATH = (
            self.folder / "matching" / "pyrano_polar_matches.json"
        )
        PYRANO_SATELLITE_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not PYRANO_SATELLITE_LIST_PATH.exists():
            print(
                "Generating polar satellite / pyrano matches and extracting pyrano values"
            )

            for satellite in self.satellites:
                if satellite.sensor == "HLS":
                    results = []
                    date_list = (
                        pd.to_datetime(
                            satellite.satellite_list[
                                "umm.TemporalExtent.RangeDateTime.BeginningDateTime"
                            ],
                            format="%Y-%m-%dT%H:%M:%S.%fZ",
                        )
                        .dt.to_pydatetime()
                        .tolist()
                    )

                    for date in date_list:
                        # Convert the date to UTC
                        date_utc = date.astimezone(timezone.utc)

                        # Convert the date to numpy.datetime64
                        timestamp_np = np.datetime64(date_utc)

                        # Ensure the dataset's time coordinate is a numpy datetime64 array
                        times_np = self.pyrano.data["time"].values.astype(
                            "datetime64[ns]"
                        )

                        # Find the index of the closest time
                        closest_index = np.abs(times_np - timestamp_np).argmin()

                        # Calculate the number of data points to extract based on the time resolution of the dataset
                        time_resolution = np.diff(times_np).min()
                        points_range = int(
                            np.timedelta64(range_in_minutes, "m") // time_resolution
                        )

                        # Calculate the start and end indices
                        start_index = max(closest_index - points_range, 0)
                        end_index = min(closest_index + points_range + 1, len(times_np))

                        # Select the data within the specified index range
                        selected_data = self.pyrano.data.isel(
                            time=slice(start_index, end_index)
                        )[variable]

                        # Apply the threshold if the variable is 'Kc' and a threshold is provided
                        if threshold is not None:
                            selected_data = selected_data.where(
                                selected_data < threshold, drop=True
                            )

                        values_within_range = selected_data.values

                        # Append the index 'date' and the extracted variable values to the results list as a dict
                        # If the variable is 'Kc' and the threshold filtering results in no data, the list will be empty.
                        results.append(
                            {"date": date_utc, variable: values_within_range.tolist()}
                        )

                        utils.save_pyrano_polar_to_json(
                            results, PYRANO_SATELLITE_LIST_PATH
                        )
        else:
            print(f"Loading pyraon-polar matche from f{PYRANO_SATELLITE_LIST_PATH}")
            results = utils.load_pyrano_polar_from_json(PYRANO_SATELLITE_LIST_PATH)

        self.pyrano_satellite_matches = results
        print(f"Filtering high variabilit dates")

        self.filtered_matching_dates = utils.filter_high_variability_high_mean_dates(
            results
        )
        print(f"Estimating solar angles")
        self.solar_angles = utils.calculate_solar_angles(
            self.filtered_matching_dates,
            float(self.pyrano.data.attrs["geospatial_lat_min"]),
            float(self.pyrano.data.attrs["geospatial_lon_min"]),
        )
        self._get_cloudnet_cloud_base_heights()

    def _download_cloudnet_data(self, cloudnet_path: str):

        unique_dates = [
            date.astimezone(pytz.utc).strftime("%Y-%m-%d")
            for date in self.filtered_matching_dates
        ]
        print(f"Downlading cloudnet products for {self.site_name} observation site")
        for date_str in unique_dates:
            glaiderutils.download_cloudnet(
                site=self.cloudnet_name , start=date_str, end=date_str, output_dir=cloudnet_path
            )
        print(f"All cloudnet products downladed {self.site_name} observation site")

    def _get_cloudnet_cloud_base_heights(self):
        CLOUDNET_PATH = self.folder / "cloudnet"
        CLOUDNET_PATH.parent.mkdir(parents=True, exist_ok=True)

        CLOUDNET_PATH_CLASSIFICATION = CLOUDNET_PATH / "Classification"
        CLOUDNET_PATH_CATEGORIZE = CLOUDNET_PATH / "Categorize"
        CLOUDNET_PATH_UPDRAFT = CLOUDNET_PATH / "Updraft"

        if self.cloudnet_name != "":
            print(f"Downloading cloudoud data")
            self._download_cloudnet_data(CLOUDNET_PATH)
            print(f"Estimating cloudoud heigth")

            self.cloud_heigth = cloud.get_mean_height(
                CLOUDNET_PATH_CLASSIFICATION, self.filtered_matching_dates
            )



