import xarray as xr
import mds
import numpy as np

def load_ens(path, sim_type):
    """
    Load in state variable ensembles.

    path: where to look look for the data
    sim_type: expt or cont
    """

    ice_fract = xr.open_dataset('{0}/ice_fract_{1}.nc'.format(path, sim_type))
    mld = xr.open_dataset('{0}/mld_{1}.nc'.format(path, sim_type))
    S = xr.open_dataset('{0}/S_{1}.nc'.format(path, sim_type))
    T = xr.open_dataset('{0}/T_{1}.nc'.format(path, sim_type))
    U = xr.open_dataset('{0}/U_{1}.nc'.format(path, sim_type))
    V = xr.open_dataset('{0}/V_{1}.nc'.format(path, sim_type))
    layers = xr.open_dataset('{0}/layers_{1}.nc'.format(path, sim_type))
    
    ice_fract_clim = xr.open_dataset('{0}/ice_fract_{1}_clim.nc'.format(path, sim_type))
    mld_clim = xr.open_dataset('{0}/mld_{1}_clim.nc'.format(path, sim_type))
    S_clim = xr.open_dataset('{0}/S_{1}_clim.nc'.format(path, sim_type))
    T_clim = xr.open_dataset('{0}/T_{1}_clim.nc'.format(path, sim_type))
    U_clim = xr.open_dataset('{0}/U_{1}_clim.nc'.format(path, sim_type))
    V_clim = xr.open_dataset('{0}/V_{1}_clim.nc'.format(path, sim_type))


    # combine into one dataset
    ensemble = xr.merge([ice_fract,
                                mld,
                                S,
                                T,
                                U,
                                V,
                                layers,
                                ice_fract_clim,
                                mld_clim,
                                S_clim,
                                T_clim,
                                U_clim,
                                V_clim,
                                ])
    return ensemble


def load_TH_diags(path, sim_type):
    """
    Load temperature diagnostics for channel model.

    path: where to look look for the data
    sim_type: expt or cont
    """

    TOTTTEND = xr.open_dataset('{0}/TOTTTEND_{1}.nc'.format(path, sim_type))
    ADVx_TH = xr.open_dataset('{0}/ADVx_TH_{1}.nc'.format(path, sim_type))
    ADVy_TH = xr.open_dataset('{0}/ADVy_TH_{1}.nc'.format(path, sim_type))
    ADVr_TH = xr.open_dataset('{0}/ADVr_TH_{1}.nc'.format(path, sim_type))
    DFxE_TH = xr.open_dataset('{0}/DFxE_TH_{1}.nc'.format(path, sim_type))
    DFyE_TH = xr.open_dataset('{0}/DFyE_TH_{1}.nc'.format(path, sim_type))
    DFrE_TH = xr.open_dataset('{0}/DFrE_TH_{1}.nc'.format(path, sim_type))
    DFrI_TH = xr.open_dataset('{0}/DFrI_TH_{1}.nc'.format(path, sim_type))
    TFLUX = xr.open_dataset('{0}/TFLUX_{1}.nc'.format(path, sim_type))
    WTHMASS = xr.open_dataset('{0}/WTHMASS_{1}.nc'.format(path, sim_type))


    TOTTTEND_clim = xr.open_dataset('{0}/TOTTTEND_{1}_clim.nc'.format(path, sim_type))
    ADVx_TH_clim = xr.open_dataset('{0}/ADVx_TH_{1}_clim.nc'.format(path, sim_type))
    ADVy_TH_clim = xr.open_dataset('{0}/ADVy_TH_{1}_clim.nc'.format(path, sim_type))
    ADVr_TH_clim = xr.open_dataset('{0}/ADVr_TH_{1}_clim.nc'.format(path, sim_type))
    DFxE_TH_clim = xr.open_dataset('{0}/DFxE_TH_{1}_clim.nc'.format(path, sim_type))
    DFyE_TH_clim = xr.open_dataset('{0}/DFyE_TH_{1}_clim.nc'.format(path, sim_type))
    DFrE_TH_clim = xr.open_dataset('{0}/DFrE_TH_{1}_clim.nc'.format(path, sim_type))
    DFrI_TH_clim = xr.open_dataset('{0}/DFrI_TH_{1}_clim.nc'.format(path, sim_type))
    TFLUX_clim = xr.open_dataset('{0}/TFLUX_{1}_clim.nc'.format(path, sim_type))
    WTHMASS_clim = xr.open_dataset('{0}/WTHMASS_{1}_clim.nc'.format(path, sim_type))

    RAC_data = mds.rdmds('{0}/40/RAC'.format(path))
    DRF_data = np.squeeze(mds.rdmds('{0}/40/DRF'.format(path)))
    hFacC_data = mds.rdmds('{0}/40/hFacC'.format(path))

    hFacC_temp = xr.DataArray(data=hFacC_data, 
             coords = {'Z':TOTTTEND.Z,
                       'YC':TOTTTEND.YC, 
                       'XC':TOTTTEND.XC},
             dims = ['Z','YC','XC'])
    hFacC = xr.Dataset(data_vars={'hFacC':hFacC_temp})


    DRF_temp = xr.DataArray(data=DRF_data, 
             coords = {'Z':TOTTTEND.Z}, 
             dims = ['Z'])
    DRF = xr.Dataset(data_vars={'DRF':DRF_temp})


    RAC_temp = xr.DataArray(data=RAC_data, 
             coords = {'YC':TOTTTEND.YC, 
                       'XC':TOTTTEND.XC},
             dims = ['YC','XC'])
    RAC = xr.Dataset(data_vars={'RAC':RAC_temp})


    # combine into one dataset
    TH_diags = xr.merge([TOTTTEND,
                                ADVx_TH,
                                ADVy_TH,
                                ADVr_TH,
                                DFxE_TH,
                                DFyE_TH,
                                DFrE_TH,
                                DFrI_TH,
                                TFLUX,
                                WTHMASS,
                                TOTTTEND_clim,
                                ADVx_TH_clim,
                                ADVy_TH_clim,
                                ADVr_TH_clim,
                                DFxE_TH_clim,
                                DFyE_TH_clim,
                                DFrE_TH_clim,
                                DFrI_TH_clim,
                                TFLUX_clim,
                                WTHMASS_clim,
                                hFacC,
                                DRF,
                                RAC])
    return TH_diags


# heat budget functions


def adv_tend(dataset, k, j, i=None, month=None):
    """Calculate advective tendency at k,j,i. If no value is passed
    for i, then the zonal mean is calculated. If no value is passed for 'month',
    then the entire time series is returned. Otherwise, just the specified
    value(s) are returned."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.ADVx_TH.shape[0])

    if i is not None:
        # a value has been passed for the x index, use it.
        cell_volume = dataset.RAC[j,i]*dataset.DRF[k]*dataset.hFacC[k,j,i]

        tend = -(dataset.ADVx_TH[month,k,j,i+1] - dataset.ADVx_TH[month,k,j,i] +
                 dataset.ADVy_TH[month,k,j+1,i] - dataset.ADVy_TH[month,k,j,i] +
                 dataset.ADVr_TH[month,k,j,i] - dataset.ADVr_TH[month,k+1,j,i])/cell_volume
    else:
        cell_volume = dataset.RAC[j,:]*dataset.DRF[k]*dataset.hFacC[k,j,:]

        tend = -(dataset.ADVy_TH[month,k,j+1,:] - dataset.ADVy_TH[month,k,j,:] +
                 dataset.ADVr_TH[month,k,j,:] - dataset.ADVr_TH[month,k+1,j,:])/cell_volume
        tend = tend.mean(dim='XC')

    return tend

def hor_adv_tend(dataset, k, j, i=None, month=None):
    """Calculate advective tendency at k,j,i. If no value is passed
    for i, then the zonal mean is calculated. If no value is passed for 'month',
    then the entire time series is returned. Otherwise, just the specified
    value(s) are returned."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.ADVx_TH.shape[0])

    if i is not None:
        # a value has been passed for the x index, use it.
        cell_volume = dataset.RAC[j,i]*dataset.DRF[k]*dataset.hFacC[k,j,i]

        tend = -(dataset.ADVx_TH[month,k,j,i+1] - dataset.ADVx_TH[month,k,j,i] +
                 dataset.ADVy_TH[month,k,j+1,i] - dataset.ADVy_TH[month,k,j,i])/cell_volume
    else:
        cell_volume = dataset.RAC[j,:]*dataset.DRF[k]*dataset.hFacC[k,j,:]

        tend = -(dataset.ADVy_TH[month,k,j+1,:] - dataset.ADVy_TH[month,k,j,:])/cell_volume
        tend = tend.mean(dim='XC')

    return tend

def vert_adv_tend(dataset, k, j, i=None, month=None):
    """Calculate advective tendency at k,j,i. If no value is passed
    for i, then the zonal mean is calculated. If no value is passed for 'month',
    then the entire time series is returned. Otherwise, just the specified
    value(s) are returned."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.ADVx_TH.shape[0])

    if i is not None:
        # a value has been passed for the x index, use it.
        cell_volume = dataset.RAC[j,i]*dataset.DRF[k]*dataset.hFacC[k,j,i]

        tend = -(dataset.ADVr_TH[month,k,j,i] - dataset.ADVr_TH[month,k+1,j,i])/cell_volume
    else:
        cell_volume = dataset.RAC[j,:]*dataset.DRF[k]*dataset.hFacC[k,j,:]

        tend = -(dataset.ADVr_TH[month,k,j,:] - dataset.ADVr_TH[month,k+1,j,:])/cell_volume
        tend = tend.mean(dim='XC')

    return tend


# diffusion term
def diff_tend(dataset, k, j, i=None, month=None):
    """Calculate the diffusive tendency at k, j, i."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.DFxE_TH.shape[0])

    if i is not None:
        cell_volume = dataset.RAC[j,i]*dataset.DRF[k]*dataset.hFacC[k,j,i]

        tend = -(dataset.DFxE_TH[month,k,j,i+1] - dataset.DFxE_TH[month,k,j,i] +
                 dataset.DFyE_TH[month,k,j+1,i] - dataset.DFyE_TH[month,k,j,i] +
                 dataset.DFrE_TH[month,k,j,i] - dataset.DFrE_TH[month,k+1,j,i] +
                 dataset.DFrI_TH[month,k,j,i] - dataset.DFrI_TH[month,k+1,j,i])/cell_volume
    else:
        cell_volume = dataset.RAC[j,:]*dataset.DRF[k]*dataset.hFacC[k,j,:]

        tend = -(dataset.DFyE_TH[month,k,j+1,:] - dataset.DFyE_TH[month,k,j,:] +
                 dataset.DFrE_TH[month,k,j,:] - dataset.DFrE_TH[month,k+1,j,:] +
                 dataset.DFrI_TH[month,k,j,:] - dataset.DFrI_TH[month,k+1,j,:])/cell_volume
        tend = tend.mean(dim='XC')

    return tend

def hor_diff_tend(dataset, k, j, i=None, month=None):
    """Calculate the diffusive tendency at k, j, i."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.DFxE_TH.shape[0])

    if i is not None:
        cell_volume = dataset.RAC[j,i]*dataset.DRF[k]*dataset.hFacC[k,j,i]

        tend = -(dataset.DFxE_TH[month,k,j,i+1] - dataset.DFxE_TH[month,k,j,i] +
                 dataset.DFyE_TH[month,k,j+1,i] - dataset.DFyE_TH[month,k,j,i])/cell_volume
    else:
        cell_volume = dataset.RAC[j,:]*dataset.DRF[k]*dataset.hFacC[k,j,:]

        tend = -(dataset.DFyE_TH[month,k,j+1,:] - dataset.DFyE_TH[month,k,j,:])/cell_volume
        tend = tend.mean(dim='XC')

    return tend

def vert_diff_tend(dataset, k, j, i=None, month=None):
    """Calculate the diffusive tendency at k, j, i."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.DFxE_TH.shape[0])

    if i is not None:
        cell_volume = dataset.RAC[j,i]*dataset.DRF[k]*dataset.hFacC[k,j,i]

        tend = -(dataset.DFrE_TH[month,k,j,i] - dataset.DFrE_TH[month,k+1,j,i] +
                 dataset.DFrI_TH[month,k,j,i] - dataset.DFrI_TH[month,k+1,j,i])/cell_volume
    else:
        cell_volume = dataset.RAC[j,:]*dataset.DRF[k]*dataset.hFacC[k,j,:]

        tend = -(dataset.DFrE_TH[month,k,j,:] - dataset.DFrE_TH[month,k+1,j,:] +
                 dataset.DFrI_TH[month,k,j,:] - dataset.DFrI_TH[month,k+1,j,:])/cell_volume
        tend = tend.mean(dim='XC')

    return tend

# Surface heat flux
def tflux_tend(dataset, k,j,i=None, month=None):
    """Contribution of surface heat flux to temperature tendency"""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.TFLUX.shape[0])

    rhoConst = 1035.
    Cp = 3994.

    if i is not None:
        if k == 0:
            tend = dataset.TFLUX[month,j,i]/(rhoConst*Cp*dataset.DRF[k]*dataset.hFacC[k,j,i])
        else:
            tend = np.zeros((dataset.TFLUX.shape[0]))
    else:
        if k == 0:
            tend = dataset.TFLUX[month,j,:]/(rhoConst*Cp*dataset.DRF[k]*dataset.hFacC[k,j,:])
            tend = tend.mean(dim='XC')
        else:
            tend = 0.*dataset.TFLUX[month,j,:].mean(dim='XC')

    return tend

# surface level correction
def surface_correction(dataset, k,j,i=None, month=None):
    """Correction for implicit linear free surface in surface level."""

    if month is not None:
        pass
    else:
        month = np.arange(dataset.WTHMASS.shape[0])


    if i is not None:
        if k == 0:
            tend = -dataset.WTHMASS[month,k,j,i]/(dataset.DRF[k]*dataset.hFacC[k,j,i])
        else:
            tend = np.zeros(dataset.WTHMASS.shape[0])
    else:
        if k == 0:
            tend = -dataset.WTHMASS[month,k,j,:]/(dataset.DRF[k]*dataset.hFacC[k,j,:])
            tend = tend.mean(dim='XC')
        else:
            tend = 0.*dataset.WTHMASS[month,k,j,:].mean(dim='XC')

    return tend

