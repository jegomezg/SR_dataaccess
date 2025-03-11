import xarray as xr
import dask
from pvlib.location import Location
from typing import List, Dict, Tuple
import numpy as np


#TODO Just an example should be replaced by either a robust schema or a real databse
SUPPORTED_PYRANO = {'PAL': {'location': {'lat':48.7126223,'lon':2.20898315},
                            'url':'BSRN-PAL.nc'},
                    'CAB':{'location': {'lat':51.9711,'lon': 4.9267},
                            'url':'BSRN-CAB.nc'},
                    'PAY':{'location': {'lat':46.815,'lon': 6.944},
                        'url':'BSRN-PAY.nc'},
                    'LIN':{'location': {'lat': 52.21,'lon': 14.122},
                        'url':'BSRN-LIN.nc'}}



#TODO extend to incorporete non supported pyrano or already implement a db that contains all of them so it can be added to directly from here
class Pyranometer():
    def __init__(self, name:str):
        self.name = name
        self.description = self._self_describe_pyrano()
        self.data = self._self_retrive_data()
        
    def _self_describe_pyrano(self):
        if self.name not in SUPPORTED_PYRANO.keys():
            raise ValueError(f'Not supported station must be one of {list(SUPPORTED_PYRANO.keys())}')
        return SUPPORTED_PYRANO[self.name]
    
    def _calculate_clear_sky(self,xds_pyrano_chunked,model:str = 'ineichen'):
        def calculate_kt(clearsky, ghi):
            return ghi / clearsky
        location = Location(xds_pyrano_chunked.latitude.item(), xds_pyrano_chunked.longitude.item(), altitude=xds_pyrano_chunked.elevation.item())

        clearsky = location.get_clearsky(xds_pyrano_chunked.indexes['time'], model=model)
        clearsky_xr = xr.Dataset.from_dataframe(clearsky)
        
        xds_pyrano_chunked['Kt'] = xr.apply_ufunc(
            calculate_kt,
            clearsky_xr['ghi'],
            xds_pyrano_chunked['GHI'],
            dask='parallelized',
            output_dtypes=[float]
        )
    
    def _self_retrive_data(self):
        xds_pyrano = xr.open_dataset(self.description['url'])
        xds_pyrano_chunked = xds_pyrano.chunk({'time': 'auto'})
        self._calculate_clear_sky(xds_pyrano_chunked)
        return xds_pyrano_chunked
    






