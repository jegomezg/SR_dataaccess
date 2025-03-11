
import os
import glob
import xarray as xr
import pandas as pd

def filter_drizzle_ice(clouds, drizzle, ice):
    def fill_and_expand(data, time_limit, height_limit):
        filled = data.ffill(dim='time', limit=time_limit).bfill(dim='time', limit=time_limit)
        expanded = filled.ffill(dim='height', limit=height_limit).bfill(dim='height', limit=height_limit)
        return expanded

    # Filter drizzle
    expand_drizzle = fill_and_expand(drizzle, time_limit=15, height_limit=10)
    clouds_out_drizzle = clouds.where(clouds.isnull() | expand_drizzle.isnull())
    fill_clouds_out_drizzle = fill_and_expand(clouds_out_drizzle, time_limit=10, height_limit=0)
    clouds_in_drizzle = clouds.where(clouds.notnull() & expand_drizzle.notnull())
    expand_clouds_in_drizzle = fill_and_expand(clouds_in_drizzle, time_limit=10, height_limit=7)
    clouds_drizzle_connect = fill_clouds_out_drizzle.where(fill_clouds_out_drizzle.notnull() & expand_clouds_in_drizzle.notnull())
    clouds_no_drizzle_time = clouds_drizzle_connect.dropna(dim='time', how='all')
    clouds_only_no_drizzle = clouds_out_drizzle.where(~clouds_out_drizzle.time.isin(clouds_no_drizzle_time.time))

    # Filter ice clouds
    expand_ice = fill_and_expand(ice, time_limit=15, height_limit=17)
    clouds_out_ice = clouds_only_no_drizzle.where(clouds_only_no_drizzle.isnull() | expand_ice.isnull())
    clouds_in_ice = clouds_only_no_drizzle.where(clouds_only_no_drizzle.notnull() & expand_ice.notnull())
    expand_clouds_in_ice = fill_and_expand(clouds_in_ice, time_limit=10, height_limit=7)
    expand_clouds_out_ice = fill_and_expand(clouds_out_ice, time_limit=10, height_limit=0)
    clouds_ice_connect = expand_clouds_out_ice.where(expand_clouds_out_ice.notnull() & expand_clouds_in_ice.notnull())
    clouds_ice_connect_time = clouds_ice_connect.dropna(dim='time', how='all')
    clouds_no_ice = clouds_out_ice.where(~clouds_out_ice.time.isin(clouds_ice_connect_time.time))
    clouds_no_ice_filled = fill_and_expand(clouds_no_ice, time_limit=2, height_limit=0)
    clouds_filtered = clouds_no_ice_filled.dropna(dim='time', how='all').dropna(dim='height', how='all')

    return clouds_filtered, clouds_no_ice

def filter_ice(clouds, ice):
    def fill_and_expand(data, time_limit, height_limit):
        filled = data.ffill(dim='time', limit=time_limit).bfill(dim='time', limit=time_limit)
        expanded = filled.ffill(dim='height', limit=height_limit).bfill(dim='height', limit=height_limit)
        return expanded

    expand_ice = fill_and_expand(ice, time_limit=15, height_limit=17)
    clouds_out_ice = clouds.where(clouds.isnull() | expand_ice.isnull())
    clouds_in_ice = clouds.where(clouds.notnull() & expand_ice.notnull())
    expand_clouds_in_ice = fill_and_expand(clouds_in_ice, time_limit=10, height_limit=7)
    expand_clouds_out_ice = fill_and_expand(clouds_out_ice, time_limit=10, height_limit=0)
    clouds_ice_connect = expand_clouds_out_ice.where(expand_clouds_out_ice.notnull() & expand_clouds_in_ice.notnull())
    clouds_ice_connect_time = clouds_ice_connect.dropna(dim='time', how='all')
    clouds_no_ice = clouds_out_ice.where(~clouds_out_ice.time.isin(clouds_ice_connect_time.time))
    clouds_no_ice_filled = fill_and_expand(clouds_no_ice, time_limit=2, height_limit=0)
    clouds_ice_filtered = clouds_no_ice_filled.dropna(dim='time', how='all').dropna(dim='height', how='all')

    return clouds_ice_filtered, clouds_no_ice
 


def get_filtered_cbh(cbh,clouds_filtered):
    cbh_clouds = cbh.where(cbh.notnull() == clouds_filtered.notnull())
    return cbh_clouds
def generate_library(path):
    '''
    Rank files based on draft abundance 'drafts' and mean velocities 'mean'
    '''
    if not os.path.exists(path):
        raise ValueError(f'{path} does not exist')

    data = [(file_path, len(xr.open_dataset(file_path).v), round(float(xr.open_dataset(file_path).v.mean().values), 4))
            for file_path in glob.glob(os.path.join(path, '*.nc'))]

    df = pd.DataFrame(data, columns=['date', 'drafts', 'mean_v'])
    
    df['drafts_normal'] = (df['drafts'] - df['drafts'].min()) / (df['drafts'].max() - df['drafts'].min())
    df['mean_v_normal'] = (df['mean_v'] - df['mean_v'].min()) / (df['mean_v'].max() - df['mean_v'].min())
    
    df['score'] = df['drafts_normal'] + df['mean_v_normal']
    
    df = df.sort_values('score', ascending=False)
    
    df['date'] = df['date'].str.split('/').str[3].str.split('_').str[0]
    
    return df