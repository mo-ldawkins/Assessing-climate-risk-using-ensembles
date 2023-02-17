"""
Code to generate plots from the paper
"""

import matplotlib.pyplot as plt
import warnings
from netCDF4 import Dataset
from scipy import sparse
import sys
import numpy as np
import glob
from matplotlib import colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

home_dir = '/data/users/ldawkins/UKCR/CodeToShare/'  # change to location of the Sup Material folder
data_dir = home_dir+'Data/'

# Humidex
t = range(15,50)
rh = range(20,100)

trh = np.empty(len(t)*len(rh)).reshape(len(rh),len(t))
for i in range(0,len(rh)):
    for j in range(0,len(t)):
        trh[i,j] = t[j] + (5 / 9) * (6.112 * (10**((7.5*t[j])/(237.7+t[j]))) * (rh[i]/100) - 10)


# Climada part

sys.path.append(home_dir+'/climada_netcdf/climada_python/')
warnings.filterwarnings('ignore')

from pandas import DataFrame
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity import Entity
from climada.engine import Impact


def impf_func(p1, p2, haz_type, int_unit, _id=1):

    def imp_arc1(hum, p1, p2):
        return 1-1/(1+(p1/hum)**(p2))

    # could add in function for other archotypes here

    imp_fun = ImpactFunc()
    imp_fun.haz_type = haz_type
    imp_fun.id = _id
    imp_fun.intensity_unit =  int_unit
    imp_fun.intensity = np.linspace(0, 100, num=100)
    imp_fun.paa = np.repeat(1, len(imp_fun.intensity))
    imp_fun.mdd = np.array([imp_arc1(hum, p1, p2) for hum in imp_fun.intensity])
    imp_fun.check()
    impf_set = ImpactFuncSet()
    impf_set.append(imp_fun)
    return impf_set

# example for UKCP BC memeber 1
ens_mem = '01'
data_source = 'UKCP_BC'
ssp = '2'
ssp_year = '2041'
warming_level='2deg'
vfn_p1=54.5
vfn_p2=-4.1
variable='humidex'
haz_type='Heatstress'
int_unit='degC'
exp_unit='Days'
save_all_imp=False
save_eai_plot=True

# hazard
netcdf_file_path = glob.glob(
                data_dir + data_source + '/*'+ens_mem+'_*humidex*'+warming_level+'*')
netcdf_file = Dataset(netcdf_file_path[0])
nyears_data = round(netcdf_file.dimensions['time'].size / 360)  # the risk calculation needs to know the number of years
nyears = (360 * nyears_data) / (360 - 102 - 8 - 25)  # 360 days per year, 102 weekend days, 8 bank holidays, 25 days AL
hazard_args = {'intensity_var': variable,
               'event_id': np.arange(len(netcdf_file.variables['time'])),
               'frequency': np.full(len(netcdf_file.variables['time']), 1/nyears),
               'haz_type': haz_type,
               'description': 'Hazard data',
               'replace_value': np.nan,
               'fraction_value': 1.0}
hazard = Hazard.from_netcdf(netcdf_file_path[0], **hazard_args)
hazard.units = '$^\circ$C'
hazard.check()
# exposure
exp_file_name = data_dir + 'UKSSPs/Employment_SSP'+ssp+'_12km_Physical.nc'
exp = Exposures.from_netcdf(exp_file_name, 'employment', int(ssp_year))
exp.value_unit = 'Number of physical jobs'
exp.set_geometry_points()
exp.check()
# vuln
imp_set = impf_func(p1=vfn_p1, p2=vfn_p2,  haz_type=haz_type, int_unit=int_unit, _id=1)

# Impact
entity = Entity()
entity.exposures = exp
entity.impact_funcs = imp_set
exp.value_unit = 'No. of days of work lost'
impact = Impact()
impact.calc(entity.exposures, entity.impact_funcs, hazard, save_mat='True')
norm = colors.LogNorm(vmin=1.0e1, vmax=1.0e6)
exp.value_unit = 'Number of physical jobs'
# calc impact for largest 'event'
impact_at_events_exp = impact._build_exp_event(4518)


######################################################################################################################
# Figure 2
fig = plt.figure(figsize=(10,10))
gs = GridSpec(nrows=3, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[2, :],projection=ccrs.PlateCarree())

ax1.set_xlabel('Temperature ($^\circ$C)')
ax1.set_ylabel('Relative Humidity (%)')
cp = ax1.contourf(t,rh,trh,levels=range(0,110))
cbar = plt.colorbar(cp,ax=ax1)
cbar.set_label('Humidex ($^\circ$C)')
ax1.title.set_text('(a)')
ax1.grid()

hazard.plot_intensity(axis=ax2, event=4518, vmin=0, vmax=110)
ax2.title.set_text('(b)')

imp_set.plot(axis=ax3)
ax3.title.set_text('(c)')
ax3.set_xlabel('Humidex Intensity ($^\circ$C)')

norm = colors.LogNorm(vmin=1.0e1, vmax=1.0e6)
exp.plot_scatter(pop_name=False, axis=ax4, norm=norm,s=11)
ax4.title.set_text('(d)')

impact_at_events_exp.plot_scatter(axis=ax5,pop_name=False, norm=norm,s=16)
ax5.title.set_text('(e)')

#plt.subplots_adjust(hspace=6.5)
plt.tight_layout()
#plt.savefig(
#      home_dir + 'Figure2_pA.png', dpi=500)
plt.show(block=True)

######################################################################################################################
# Figure 3
ens_mem = '01'
data_source = 'UKCP_BC'
ssp = '2'
ssp_year = '2020'
warming_level='1998'
vfn_p1=54.5
vfn_p2=-4.1
variable='humidex'
haz_type='Heatstress'
int_unit='degC'
exp_unit='Days'
save_all_imp=False
save_eai_plot=True

# calc EAI from member 1
# hazard
netcdf_file_path = glob.glob(
                data_dir + data_source + '/*'+ens_mem+'_*humidex*'+warming_level+'*')
netcdf_file = Dataset(netcdf_file_path[0])
nyears_data = round(netcdf_file.dimensions['time'].size / 360)  # the risk calculation needs to know the number of years
nyears = (360 * nyears_data) / (360 - 102 - 8 - 25)  # 360 days per year, 102 weekend days, 8 bank holidays, 25 days AL
hazard_args = {'intensity_var': variable,
               'event_id': np.arange(len(netcdf_file.variables['time'])),
               'frequency': np.full(len(netcdf_file.variables['time']), 1/nyears),
               'haz_type': haz_type,
               'description': 'Hazard data',
               'replace_value': np.nan,
               'fraction_value': 1.0}
hazard = Hazard.from_netcdf(netcdf_file_path[0], **hazard_args)
hazard.units = '$^\circ$C'
hazard.check()
# exposure
exp_file_name = data_dir + '/UKSSPs/Employment_SSP'+ssp+'_12km_Physical.nc'
exp = Exposures.from_netcdf(exp_file_name, 'employment', int(ssp_year))
exp.value_unit = 'Number of physical jobs'
exp.set_geometry_points()
exp.check()
# vuln
imp_set = impf_func(p1=vfn_p1, p2=vfn_p2,  haz_type=haz_type, int_unit=int_unit, _id=1)

# Impact
entity = Entity()
entity.exposures = exp
entity.impact_funcs = imp_set
exp.value_unit = 'No. of days of work lost'
impact = Impact()
impact.calc(entity.exposures, entity.impact_funcs, hazard, save_mat='True')
norm = colors.LogNorm(vmin=1.0e1, vmax=1.0e6)
exp.value_unit = 'Number of physical jobs'


# calc ensemble mean based on all members for plot (b)
warming_level='current'
ssp='2'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
ds = xr.open_dataset(
    data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
df = ds.to_dataframe()
df['member'] = ens_mem
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = xr.open_dataset(
       data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
    df2 = ds.to_dataframe()
    df2['member'] = ens_mem
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean = df.groupby(['exposure_longitude','exposure_latitude']).mean()
ens_mean.loc[ens_mean['annual_impact']==0] = np.nan
ens_mean = ens_mean.reset_index(level=0)
ens_mean = ens_mean.reset_index(level=0)

# Find spatially aggregated EAI for each ensemble member
spaceagg = df[['annual_impact','member']].groupby(['member']).sum()


# Load in for observed
ds = xr.open_dataset(
    data_dir + '/Obs/expected_annual_impact_data_Obs_ensnan_WLcurrent_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
obs_eai = ds.to_dataframe()
#obs_eai.loc[obs_eai['annual_impact']==0] = np.nan
obs_eai = obs_eai.reset_index(level=0)
obs_eai = obs_eai.reset_index(level=0)

obs_spaceagg = obs_eai[['annual_impact']].sum()

bias_eai = pd.merge(obs_eai, ens_mean, how = 'left', left_on = ['exposure_longitude','exposure_latitude'],
                    right_on = ['exposure_longitude','exposure_latitude'])

bias_eai['bias'] = (bias_eai['annual_impact_y'] - bias_eai['annual_impact_x'])/bias_eai['annual_impact_x']

# plot figure
import seaborn as sns
#plt.figure(figsize=(9.5,9))
fig, ax = plt.subplots(2, 2, figsize=(9,7))
ax1 = plt.subplot(2,2,1,projection=ccrs.PlateCarree())
ax2 = plt.subplot(2,2,2,projection=ccrs.PlateCarree())
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4,projection=ccrs.PlateCarree())
plt.subplots_adjust(wspace = 0.33 )

impact.plot_scatter_eai_exposure(axis=ax1,pop_name=False,norm=norm,s=16)
ax1.title.set_text('(a)')

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(ens_mean['exposure_longitude'],ens_mean['exposure_latitude'],c=ens_mean['annual_impact'],norm=norm,s=12,cmap='magma_r')
cbar = plt.colorbar(cp,ax=ax2,shrink=0.8)
cbar.set_label('Value (No. of days of work lost)')
ax2.title.set_text('(b)')
ax2.coastlines()

ax3.set_xlabel('Spatially aggregated risk \n (No. of days of work lost)')
ax3.set_ylabel('Frequency')
ax3.set_xlim(1e7,1.1e7)
ax3.hist(spaceagg['annual_impact'],density=False,color='cornflowerblue')
ax3.vlines(obs_spaceagg[0],ymin=0,ymax=3,linestyles='solid',colors='purple',linewidth=2)
ax3.vlines(spaceagg.mean()[0],ymin=0,ymax=3,linestyles='solid',colors='blue')
#sns.kdeplot(data=spaceagg['annual_impact'],ax=ax3)
ax3.title.set_text('(c)')
legend_elements = [Patch(facecolor='cornflowerblue', edgecolor='cornflowerblue',label='UKCP18 ensemble'),
                    Line2D([0], [0], color='blue', linestyle='solid', label='Ensemble mean'),
                   Line2D([0], [0], color='purple', linestyle='solid', label='Observed',linewidth=2)]
ax3.legend(handles=legend_elements,loc='upper right')

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
cp = ax4.scatter(bias_eai['exposure_longitude'],bias_eai['exposure_latitude'],c=bias_eai['bias'],vmin=-0.1,vmax=0.1,s=10,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax4,shrink=1)
cbar.set_label('Relative Bias [(Ensemble mean - Observed)/Observed]')
ax4.title.set_text('(d)')
ax4.coastlines()
plt.tight_layout()
plt.savefig(
       home_dir + 'Figure3_pA.png', dpi=500)
plt.show(block=True)

###############################################################################################
# Fig 4 - Future risk
# 2deg
warming_level='2deg'
ssp='2'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
ds = xr.open_dataset(
    data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
df = ds.to_dataframe()
df['member'] = ens_mem
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = xr.open_dataset(
        data_dir  + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
    df2 = ds.to_dataframe()
    df2['member'] = ens_mem
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_2deg = df.groupby(['exposure_longitude','exposure_latitude']).mean()
ens_mean_2deg.loc[ens_mean_2deg['annual_impact']==0] = np.nan
ens_mean_2deg = ens_mean_2deg.reset_index(level=0)
ens_mean_2deg = ens_mean_2deg.reset_index(level=0)
spaceagg_2deg = df[['annual_impact','member']].groupby(['member']).sum()

# 4deg
warming_level='4deg'
ssp='2'
ens_mems = ['01', '04', '05', '06', '07', '09', '11', '12', '13']
ens_mem = ens_mems[0]
ds = xr.open_dataset(
    data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
df = ds.to_dataframe()
df['member'] = ens_mem
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = xr.open_dataset(
        data_dir + data_source + '/expected_annual_impact_data_' + data_source + '_ens' + ens_mem + '_WL' + warming_level + '_SSP' + ssp + '_vp1=' + str(vfn_p1) + '_vp2=' + str(vfn_p2) + '.nc')
    df2 = ds.to_dataframe()
    df2['member'] = ens_mem
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_4deg = df.groupby(['exposure_longitude','exposure_latitude']).mean()
ens_mean_4deg.loc[ens_mean_4deg['annual_impact']==0] = np.nan
ens_mean_4deg = ens_mean_4deg.reset_index(level=0)
ens_mean_4deg = ens_mean_4deg.reset_index(level=0)
spaceagg_4deg = df[['annual_impact','member']].groupby(['member']).sum()

#plot
fig = plt.figure(figsize=(10,8))
gs = GridSpec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[1, :])

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(ens_mean_2deg['exposure_longitude'],ens_mean_2deg['exposure_latitude'],c=ens_mean_2deg['annual_impact'],norm=norm,s=12,cmap='magma_r')
cbar = plt.colorbar(cp,ax=ax1,shrink=0.8)
cbar.set_label('Value (No. of days of work lost)')
ax1.title.set_text('(a)')
ax1.coastlines()

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(ens_mean_4deg['exposure_longitude'],ens_mean_4deg['exposure_latitude'],c=ens_mean_4deg['annual_impact'],norm=norm,s=12,cmap='magma_r')
cbar = plt.colorbar(cp,ax=ax2,shrink=0.8)
cbar.set_label('Value (No. of days of work lost)')
ax2.title.set_text('(b)')
ax2.coastlines()

binwidthc = 3e5
binsusec = np.arange(6e6, 4e7 + binwidthc, binwidthc)
binwidth = 5e5
binsuse = np.arange(6e6, 4e7 + binwidth, binwidth)
ax3.set_xlabel('Spatially aggregated risk \n (No. of days of work lost)')
ax3.set_ylabel('Frequency')
ax3.set_xlim(6e6,4e7)
ax3.hist(spaceagg['annual_impact'],density=False,color='cornflowerblue',label='Current climate (recent past)',bins=binsusec)
ax3.hist(spaceagg_2deg['annual_impact'],density=False,color='palegreen',label='2oC warming level',bins=binsuse)
ax3.hist(spaceagg_4deg['annual_impact'],density=False,color='darksalmon',label='4oC warming level',bins=binsuse)
ax3.legend(loc='upper right')
ax3.title.set_text('(c)')

plt.tight_layout()
plt.savefig(
       home_dir + 'Figure4_pA.png', dpi=500)
plt.show(block=True)

######################################################################################################################

# Figures 5 made in R - code can be provided on request

######################################################################################################################
# Fig 6 - the new improved version of Fig 4
# EAI lower, mean and upper CI + histogram

# GAM samples
# 2deg
data = Dataset(data_dir + '/UKCP_BC/GAMsamples_expected_annual_impact_data_UKCP_BC_WL2deg_SSP2_vp1=54.5_vp2=-4.1.nc')
allEAI = np.array(data.variables['sim_annual_impact'])
allEAI[np.where(allEAI > 9e30)] = np.nan
allEAI = 10 ** allEAI - 1
EAI_lower = np.quantile(allEAI,0.025,axis=2).reshape(110*83)
EAI_mean = np.nanmean(allEAI,axis=2).reshape(110*83)
EAI_upper = np.quantile(allEAI,0.975,axis=2).reshape(110*83)
EAI_lower_diff = EAI_lower - EAI_mean
EAI_upper_diff = EAI_upper - EAI_mean

# 4deg
data = Dataset(data_dir + '/UKCP_BC/GAMsamples_expected_annual_impact_data_UKCP_BC_WL4deg_SSP2_vp1=54.5_vp2=-4.1.nc')
allEAI = np.array(data.variables['sim_annual_impact'])
allEAI[np.where(allEAI > 9e30)] = np.nan
allEAI = 10 ** allEAI - 1
EAI_lower = np.quantile(allEAI,0.025,axis=2).reshape(110*83)
EAI_mean_4deg = np.nanmean(allEAI,axis=2).reshape(110*83)
EAI_upper = np.quantile(allEAI,0.975,axis=2).reshape(110*83)
EAI_lower_diff_4deg = EAI_lower - EAI_mean_4deg
EAI_upper_diff_4deg = EAI_upper - EAI_mean_4deg

# for histogram
#hist
ds = xr.open_dataset(
    data_dir + '/UKCP_BC/GAMsamples_expected_annual_impact_data_UKCP_BC_WLcurrent_SSP2_vp1=54.5_vp2=-4.1.nc')
df = ds.to_dataframe()
df['sim_annual_impact'] = df['sim_annual_impact'].replace(9.969210e+36, 0)
df['sim_annual_impact'] = 10 ** df['sim_annual_impact'] - 1
spaceagg = df[['sim_annual_impact','simulation_number']].groupby(['simulation_number']).sum()

#2deg
ds = xr.open_dataset(
    data_dir + '/UKCP_BC/GAMsamples_expected_annual_impact_data_UKCP_BC_WL2deg_SSP2_vp1=54.5_vp2=-4.1.nc')
df = ds.to_dataframe()
df['sim_annual_impact'] = df['sim_annual_impact'].replace(9.969210e+36, 0)
df['sim_annual_impact'] = 10 ** df['sim_annual_impact'] - 1
spaceagg_2deg = df[['sim_annual_impact','simulation_number']].groupby(['simulation_number']).sum()

#4deg
ds = xr.open_dataset(
    data_dir + '/UKCP_BC/GAMsamples_expected_annual_impact_data_UKCP_BC_WL4deg_SSP2_vp1=54.5_vp2=-4.1.nc')
df = ds.to_dataframe()
df['sim_annual_impact'] = df['sim_annual_impact'].replace(9.969210e+36, 0)
df['sim_annual_impact'] = 10 ** df['sim_annual_impact'] - 1
spaceagg_4deg = df[['sim_annual_impact','simulation_number']].groupby(['simulation_number']).sum()


lon = np.array(data.variables['exposure_longitude']).reshape(110*83)
lat = np.array(data.variables['exposure_latitude']).reshape(110*83)


fig = plt.figure(figsize=(12,12))
gs = GridSpec(nrows=3, ncols=3)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[0, 2],projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
ax6 = fig.add_subplot(gs[1, 2],projection=ccrs.PlateCarree())
ax7 = fig.add_subplot(gs[2, :])

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
#cp = ax1.scatter(lon,lat,c=EAI_lower,norm=norm,s=12,cmap='magma_r')
cp = ax1.scatter(lon,lat,c=EAI_lower_diff,s=12,vmin=-5e4,vmax=5e4,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax1,shrink=0.5)
cbar.set_label('EAI (lower bound of 95% CI - mean)')
ax1.title.set_text('(a)')
ax1.coastlines()

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(lon,lat,c=EAI_mean,norm=norm,s=12,cmap='magma_r')
cbar = plt.colorbar(cp,ax=ax2,shrink=0.5)
cbar.set_label('EAI (mean)')
ax2.title.set_text('(b)')
ax2.coastlines()

ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
#cp = ax3.scatter(lon,lat,c=EAI_upper,norm=norm,s=12,cmap='magma_r')
cp = ax3.scatter(lon,lat,c=EAI_upper_diff,s=12,vmin=-5e4,vmax=5e4,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax3,shrink=0.5)
cbar.set_label('EAI (upper bound of 95% CI - mean)')
ax3.title.set_text('(c)')
ax3.coastlines()

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
#cp = ax1.scatter(lon,lat,c=EAI_lower,norm=norm,s=12,cmap='magma_r')
cp = ax4.scatter(lon,lat,c=EAI_lower_diff_4deg,s=12,vmin=-5e4,vmax=5e4,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax4,shrink=0.5)
cbar.set_label('EAI (lower bound of 95% CI - mean)')
ax4.title.set_text('(d)')
ax4.coastlines()

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(lon,lat,c=EAI_mean_4deg,norm=norm,s=12,cmap='magma_r')
cbar = plt.colorbar(cp,ax=ax5,shrink=0.5)
cbar.set_label('EAI (mean)')
ax5.title.set_text('(e)')
ax5.coastlines()

ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
#cp = ax3.scatter(lon,lat,c=EAI_upper,norm=norm,s=12,cmap='magma_r')
cp = ax6.scatter(lon,lat,c=EAI_upper_diff_4deg,s=12,vmin=-5e4,vmax=5e4,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax6,shrink=0.5)
cbar.set_label('EAI (upper bound of 95% CI - mean)')
ax6.title.set_text('(f)')
ax6.coastlines()


binwidthc = 3e5
binsusec = np.arange(6e6, 4e7 + binwidthc, binwidthc)
binwidth = 5e5
binsuse = np.arange(6e6, 4e7 + binwidth, binwidth)
binwidth = 5e5
binsuse = np.arange(6e6, 4e7 + binwidth, binwidth)

ax7.set_xlabel('Spatially aggregated risk \n (No. of days of work lost)')
ax7.set_ylabel('Frequency (GAM samples)')
ax7.set_xlim(6e6,4e7)
ax7.hist(spaceagg_sim['sim_annual_impact'],density=False,color='darkblue',label='Current climate (recent past): GAM',alpha=0.5,bins=binsusec)
ax7.hist(spaceagg_2deg_sim['sim_annual_impact'],density=False,color='green',label='2oC warming level: GAM',alpha=0.5,bins=binsuse)
ax7.hist(spaceagg_4deg_sim['sim_annual_impact'],density=False,color='red',label='4oC warming level: GAM',alpha=0.5,bins=binsuse)


ax8 = ax7.twinx()
ax8.hist(spaceagg['annual_impact'],density=False,color='cornflowerblue',label='Current climate (recent past): UKCP18 Ensemble',alpha=0.6,bins=binsusec)
ax8.hist(spaceagg_2deg['annual_impact'],density=False,color='palegreen',label='2oC warming level: UKCP18 Ensemble',alpha=0.6,bins=binsuse)
ax8.hist(spaceagg_4deg['annual_impact'],density=False,color='darksalmon',label='4oC warming level: UKCP18 Ensemble',alpha=0.6,bins=binsuse)
ax8.set_ylabel('Frequency (UKCP18 Ensemble)')

ax7.legend(loc='upper right')
ax8.legend(loc='upper center')
ax7.title.set_text('(g)')


plt.tight_layout()
plt.savefig(
       home_dir + 'Figure6_pA.png', dpi=500)
plt.show(block=True)

######################################################################################################################

# Figures 7 made in R - code can be provided on request

######################################################################################################################
