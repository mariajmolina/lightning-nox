from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy
import cartopy.crs as ccrs


variables = [
    'precon',
    'energy_f',
    'energy_g',
    'flashes',
    'groups',
    'cape',
    'mf_cbase',
    'flashes',
    'pretot',
    'iwc_440',
    'pblh',
    'p_ctop',
    'p_cbase',
    'cnv_mfc_440',
    'z_cbase',
    'cldht',
    't_sfc',
    'cldfrac_conv_440',
    'cldfrac_ls_440',
    'l_cbase'
]


def open_and_preprocess(filename, variables):

    # open file
    ds = xr.open_dataset(filename, mask_and_scale=True)
    
    # find the corresponding variable string
    for var in variables:
        try:
            ds[var]
            break
        except KeyError:
            continue

    # create datetime range
    # automatically grab month and year from filename string
    datetime_range = pd.date_range(
        start=f"{filename.split('.r180W')[0][-6:-2]}-{filename.split('.r180W')[0][-2:]}-01", 
        freq="1H", 
        periods=len(ds.Days)*len(ds.Hours)
    )
    
    # add one hour to datetime array due to ds format
    datetime_range = datetime_range + timedelta(hours=1)
    
    # change fill values to nans
    # create new ds with coords including cleaned up time
    # change lat/lon/time from variables to coords
    newds = ds.where(
        ds[var]!=1.e+15,np.nan).assign_coords(
        coords=dict(
            Longitudes=ds.longitude,
            Latitudes=ds.latitude,
            Datetime=datetime_range,
        )
    ).drop_vars(
        names=["longitude","latitude","time"]
    ).stack(Datetime=["Days","Hours"])[var].assign_attrs(
        missing_value=np.nan).to_dataset(
        name=var).drop_vars(["Datetime","Days","Hours"]).assign_coords(
        coords=dict(
            Datetime=datetime_range,
        )
    )
    
    # return the new data array
    return newds[var]


def gridlines(axis):
    gridliner = axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  
    gridliner.xlines = False


def case_study_plotting(ds_tmp,
                        CS1_tmp,
                        CS2_tmp,
                        lon_range,
                        lat_range,  
                        date_time,
                        vmin=None,
                        vmax=None,
                        cmap='plasma',
                        cnt_levels_1=None, 
                        color_1='white',
                        legend_label_1=None,
                        cnt_levels_2=None,
                        color_2='blue',
                        legend_label_2=None,
                        cbar_label=None,
                        legend_facecolor='red',
                        fig_title=None,
                        save_string=None,
                       ):
    """
    Automates plotting of case studies for the lightning study.
    
    Args:
        ds_tmp (array): NASA GEOS data array
        CS1_tmp (array): GLM data array 1
        CS2_tmp (array): GLM data array 2
        lon_range (floats): written as slice(lon0,lon1)
        lat_range (floats): written as slice(lat0,lat1)
        date_time (string): written as '2019-07-05T20:00:00.000000000'
        vmin=None (float): minimum NASA GEOS data range. Defaults to None.
        vmax=None (float): maximum NASA GEOS data range. Defaults to None.
        cmap='plasma' (string): NASA GEOS colormap option. Defaults to plasma.
        cnt_levels_1=None (list of values): contour levels for GLM variable 1.
        color_1='white' (string): color for GLM variable 1.
        legend_label_1=None (string): legend label for GLM variable 1.
        cnt_levels_2=None (list of values): contour levels for GLM variable 1.
        color_2='blue' (string): color for GLM variable 2.
        legend_label_2=None (string): legend label for GLM variable 2.
        cbar_label=None (string): colorbar label for NASA GEOS variable.
        legend_facecolor='red' (string): facecolor for legend
        fig_title=None (string): title for figure
        save_string=None (string): save filename and directory for figure. Defaults to None.
    """
    # figure creation
    fig, axis = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Mercator()))

    # plotting
    ds = ds_tmp.sel(
        Latitudes=lat_range,
        Longitudes=lon_range,
        Datetime=date_time,
    )

    PC = axis.pcolormesh(
        ds.Longitudes, 
        ds.Latitudes, 
        ds.data, 
        vmin=vmin, 
        vmax=vmax, 
        cmap=cmap, 
        transform=ccrs.PlateCarree(),
    )

    gridlines(axis)

    plt.colorbar(PC, ax=axis, extend='max', label=cbar_label)

    # plotting the group flashes
    CS = CS1_tmp.sel(
        Latitudes=lat_range,
        Longitudes=lon_range,
        Datetime=date_time,
    )

    cs = axis.contour(
        CS.Longitudes, 
        CS.Latitudes, 
        CS.data, 
        levels=cnt_levels_1, 
        colors=color_1, 
        transform=ccrs.PlateCarree(),
    )

    # labeling the contour
    axis.clabel(cs, fontsize=7.5) 

    # plotting the individual flashes
    CS = CS2_tmp.sel(
        Latitudes=lat_range,
        Longitudes=lon_range,
        Datetime=date_time,
    )

    cs = axis.contour(
        CS.Longitudes, 
        CS.Latitudes, 
        CS.data, 
        levels=cnt_levels_2,
        colors=color_2, 
        linestyles="dashed", 
        transform=ccrs.PlateCarree(),
    )
    
    axis.clabel(
        cs,
        inline=True,
        fontsize=7.5
    )

    axis.coastlines()
    axis.add_feature(cartopy.feature.STATES)
    
    custom_lines = [
        Line2D([0], [0], color=color_1, linestyle='-'),
        Line2D([0], [0], color=color_2, linestyle='--')
                   ]
    
    axis.legend(
        custom_lines, 
        [legend_label_1, legend_label_2], 
        framealpha=1., 
        facecolor=legend_facecolor)

    axis.set_title(fig_title)
    
    if save_string is not None:
        plt.savefig(fname=save_string, dpi=200)
    
    plt.show()
    plt.close()
    
    return