"""
# For a given data source, associated ensemble members, warming level, SSP,
# SSP year associated with the warming level and vulnerability function parameters
# reads in Expected Annual Impact (as
# output from the climada_risk_calc script) from all ensemble members and creates a data
# frame of this combined with long, lat, orography and exposure (number of jobs
# as used in risk calculation). The code then imports the R ‘mgcv’ package and
# uses this to fit a spatially varying generalised additive model for EAI with
# the superior functionality of this package (better than gam packages in
# Python) but within Python. Samples are then taken from the posterior predictive
# distribution of EAI based on this model. These are then saved out as a netcdf file.
"""

import pandas as pd
import xarray as xr
import numpy as np
from netCDF4 import Dataset
import cftime
import sys
pd.set_option('display.max_columns', None)

import os
os.environ['R_HOME'] = '/home/h01/ldawkins/.conda/envs/climada_env_2022/lib/R'

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()
r_mgcv = importr('mgcv')
base = importr('base')
stats = importr('stats')

home_dir = '/data/users/ldawkins/UKCR/CodeToShare/'  # change to location of the Sup Material folder
data_dir = home_dir+'Data/'

def applyGAM(data_source,ens_mems,warming_level,ssp,ssp_year,vp1,vp2,SA=False):

    # Make dataframe for GAM fitting
    # load in risk (EAI)
    ens_mem = ens_mems[0]
    if SA == True:
        ds = xr.open_dataset(
            data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(
                vp1) + '_vp2=' + str(vp2) + '.nc')
    else:
        ds = xr.open_dataset(data_dir+data_source+'/expected_annual_impact_data_'+data_source+'_ens'+ens_mem+'_WL'+warming_level+'_SSP'+ssp+'.nc')
    df = ds.to_dataframe()
    df['member'] = ens_mem
    for ens_mem in ens_mems[1:len(ens_mems)]:
        if SA == True:
            ds = xr.open_dataset(
                data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(
                    vp1) + '_vp2=' + str(vp2) + '.nc')
        else:
            ds = xr.open_dataset(
                data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '.nc')
        df2 = ds.to_dataframe()
        df2['member'] = ens_mem
        df = pd.concat([df,df2])
    df['annual_impact'] = np.log10(df['annual_impact']+1)
    # orography
    orog_netcdf = Dataset(data_dir + 'UKCP_RCM_orog.nc')
    orog = orog_netcdf.variables['surface_altitude'][...].ravel()
    df['orog'] = np.tile(np.array(orog),len(ens_mems))
    df['orog'] = np.log(df['orog']+5)
    # population
    exposure_netcdf = Dataset(data_dir +'/UKSSPs/Employment_SSP'+ssp+'_12km_Physical.nc')
    units = getattr(exposure_netcdf['time'], 'units')
    calendar = getattr(exposure_netcdf['time'], 'calendar')
    dates = cftime.num2date(exposure_netcdf.variables['time'][:], units, calendar)
    year_to_index = {k.timetuple().tm_year: v for v, k in enumerate(dates)}
    index = year_to_index[int(ssp_year)]
    df['pop'] = np.tile(exposure_netcdf.variables['employment'][index][...].ravel(),len(ens_mems))
    df['pop'] = np.log10(df['pop']+1)
    # remove non-land grid cells
    df = df[df['pop'].notna()]
    df['member'] = df['member'].astype('category')

    # to test on less data
    #df = df.iloc[0:(1711 * 2), :]

    # convert to R dataframe
    R_df = ro.conversion.py2rpy(df)

    # define GAM functions for location and scale
    modparams = []
    modparams.append("annual_impact ~ ti(exposure_longitude, exposure_latitude, k=15,bs='tp') +"
                     " ti(exposure_longitude,exposure_latitude,orog,d=c(2,1),bs=c('tp','tp'),k=c(15,6)) + ti(orog,k=6) +"
                     " ti(pop,k=6) + ti(member,bs='re')")
    modparams.append("~ 1")

    # fit GAM
    gamFit = r_mgcv.gam([ro.Formula(modparams[0]),ro.Formula(modparams[1])], data=R_df, family='gaulss', method='REML')

    # Generates samples of EAI from the GAM
    # Define new dataframe with location data for predicitions
    nloc = len(df['exposure_longitude'].unique())
    newdata = df.iloc[0:nloc]
    newd = newdata[['exposure_latitude','exposure_longitude','orog','pop','member']]

    # extract parameters and the covariance matrix of the parameters
    coefs = stats.coef(gamFit)
    Vc=np.asmatrix(gamFit.rx2('Vc'))

    # define number of knots
    nknots = np.array([len(coefs)-1,1]) # the last one is the scale, all others are associated with the mean
    # define no of simulations required
    nsims = 1000

    # define coefs and Vc as R vector/matrix
    coefs_R = ro.vectors.FloatVector(coefs)
    Vc_R = ro.r.matrix(Vc, nrow=len(coefs), ncol=len(coefs))

    # sample parameters from MVN posterior
    betas = r_mgcv.rmvn(nsims,coefs_R,Vc_R)

    # extract the lpmatrix [vector of linear predictor values (minus any offest) at the supplied covariate values, when applied to the model coefficient vector]
    X = stats.predict(gamFit, newdata=newd, type="lpmatrix", exclude="ti(member)")

    # extract the random effect SD
    ro.globalenv['Model'] = gamFit
    vcomp = ro.r('gam.vcomp(Model)')
    rownames = list(ro.r('row.names(gam.vcomp(Model))'))
    whichone = list()
    for i in range(0,len(rownames)):
        whichone.append(rownames[i] == 'ti(member)')
    ind = whichone.index(True)
    RE_sd = vcomp[ind, 0]

    msamp = stats.rnorm(nsims, 0, RE_sd)
    msamp_mat = np.tile(msamp, (newd.shape[0], 1))

    # calculate the GAM mean and sd at each location
    Mean = np.dot(X[:,0:int(nknots[0])],np.transpose(betas[:,0:int(nknots[0])])) + msamp_mat
    LPsd = np.dot(X[:,int(nknots[0]):(int(nknots[0])+int(nknots[1]))],np.transpose(betas[:,int(nknots[0]):int(nknots[0])+int(nknots[1])]))
    Sd = np.exp(LPsd) + 0.01

    # simulate from predictive distribution
    preds = np.empty(newd.shape[0]*nsims).reshape(newd.shape[0],nsims)
    for i in range(0,newd.shape[0]):
        for j in range(0,nsims):
            preds[i,j] = stats.rnorm(1, mean = Mean[i,j], sd = Sd[i,j])

    return preds

def preds_to_netcdf(preds, shape, file_name):

    # convert to ncdf structure + save

    #load in orog info to use when building output
    netcdf_orog = Dataset(data_dir + 'UKCP_RCM_orog.nc')
    data = netcdf_orog.variables['surface_altitude'][...].ravel()
    lon = netcdf_orog.variables['longitude'][...].ravel()
    lat = netcdf_orog.variables['latitude'][...].ravel()
    lat = np.reshape(lat, shape[0:2])
    lon = np.reshape(lon, shape[0:2])
    ind = np.where(data.mask == False)[0]

    # make new file
    netcdf_file = Dataset(file_name, "w", format="NETCDF4")

    netcdf_file.createDimension("latitude", shape[0])
    netcdf_file.createDimension("longitude", shape[1])
    latitude = netcdf_file.createVariable("exposure_latitude", "f4", ("latitude", "longitude"))
    longitude = netcdf_file.createVariable("exposure_longitude", "f4", ("latitude", "longitude"))
    latitude[:] = lat
    longitude[:] = lon

    netcdf_file.createDimension("simulation", shape[2])
    simulation = netcdf_file.createVariable("simulation_number", "f4", "simulation")
    simulation[:] = range(1,shape[2]+1)
    sim_annual_impact = netcdf_file.createVariable("sim_annual_impact", "f4", ("latitude", "longitude", "simulation"))

    sim_eai = np.zeros(data.shape[0] * shape[2]).reshape(shape)
    for j in range(0, shape[2]):
        use = preds[:, j]
        new = data
        new[ind] = use  # over write non-masked part with preds
        new = np.reshape(new, shape[0:2])
        sim_eai[:, :, j] = new

    sim_annual_impact[:] = sim_eai

    annual_impact_meta = {"data_source": "GAM sample from model fitted to Annual Impact Data generated by climada"}
    netcdf_file.setncatts(annual_impact_meta)

    netcdf_file.close()


# function to find part of a string - used to define which UKCORDEX ens members are included in each warming level fit
def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

# ----------------------------------------------------------------------------------------------------

# apply to all combos of
data_source = 'UKCP_BC'
ssp = '2'
warming_level= ['current','2deg','4deg']
ssp_year = ['2020','2041','2084']

vp1=54.5
vp2=-4.1

if data_source == 'UKCP_BC' and warming_level == '4deg':
    ens_mems = ['01', '04', '05', '06', '07', '09', '11', '12', '13']
else:
    ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

pred_samples = applyGAM(data_source,ens_mems,warming_level,ssp,ssp_year, vp1, vp2, SA=True)

shape = [110,83,1000]

preds_to_netcdf(pred_samples, shape, file_name= data_dir +data_source+'/GAMsamples_expected_annual_impact_data_'+data_source+'_WL'+warming_level+'_SSP'+ssp+'_vp1='+vp1+'_vp2='+vp2+'.nc')



