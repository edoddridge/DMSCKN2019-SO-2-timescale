import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


from channel_funcs import load_ens, load_TH_diags

# heat budget functions
from channel_funcs import adv_tend, hor_adv_tend, vert_adv_tend
from channel_funcs import diff_tend, hor_diff_tend, vert_diff_tend
from channel_funcs import tflux_tend, surface_correction


SAM_uTq_expt_TH_diags = load_TH_diags('../SO_3d/CORE/SAM_utq/one_lobe/', 'expt')

SAM_uTq_cont_TH_diags = load_TH_diags('../SO_3d/CORE/SAM_utq/one_lobe/', 'cont')


k = 10
j = 290
jrange = 70

adv_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
hor_adv_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
vert_adv_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
diff_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
hor_diff_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
vert_diff_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
tflux_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
surf_corr_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
TOTTTEND_cont = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))

adv_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
hor_adv_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
vert_adv_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
diff_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
hor_diff_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
vert_diff_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
tflux_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
surf_corr_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))
TOTTTEND_expt = np.zeros((SAM_uTq_cont.time_mnthly.shape[0], jrange))

for n in range(jrange):

    adv_cont[:,n] = adv_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    hor_adv_cont[:,n] = hor_adv_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    vert_adv_cont[:,n] = vert_adv_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    diff_cont[:,n] = diff_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    hor_diff_cont[:,n] = hor_diff_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    vert_diff_cont[:,n] = vert_diff_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    tflux_cont[:,n] = tflux_tend(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    surf_corr_cont[:,n] = surface_correction(SAM_uTq_cont_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    TOTTTEND_cont[:,n] = SAM_uTq_cont_TH_diags.TOTTTEND[:,k,j+n,:].mean(dim=['XC'])/86400

    adv_expt[:,n] = adv_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    hor_adv_expt[:,n] = hor_adv_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    vert_adv_expt[:,n] = vert_adv_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    diff_expt[:,n] = diff_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    hor_diff_expt[:,n] = hor_diff_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    vert_diff_expt[:,n] = vert_diff_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    tflux_expt[:,n] = tflux_tend(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    surf_corr_expt[:,n] = surface_correction(SAM_uTq_expt_TH_diags, k, j+n)#, month=month)#.mean(dim='YC')
    TOTTTEND_expt[:,n] = SAM_uTq_expt_TH_diags.TOTTTEND[:,k,j+n,:].mean(dim=['XC'])/86400 

# take averages over the box
adv_cont = adv_cont.mean(axis=1)
hor_adv_cont = hor_adv_cont.mean(axis=1)
vert_adv_cont = vert_adv_cont.mean(axis=1)
diff_cont = diff_cont.mean(axis=1)
hor_diff_cont = hor_diff_cont.mean(axis=1)
vert_diff_cont = vert_diff_cont.mean(axis=1)
tflux_cont = tflux_cont.mean(axis=1)
surf_corr_cont = surf_corr_cont.mean(axis=1)
TOTTTEND_cont = TOTTTEND_cont.mean(axis=1)

adv_expt = adv_expt.mean(axis=1)
hor_adv_expt = hor_adv_expt.mean(axis=1)
vert_adv_expt = vert_adv_expt.mean(axis=1)
diff_expt = diff_expt.mean(axis=1)
hor_diff_expt = hor_diff_expt.mean(axis=1)
vert_diff_expt = vert_diff_expt.mean(axis=1)
tflux_expt = tflux_expt.mean(axis=1)
surf_corr_expt = surf_corr_expt.mean(axis=1)
TOTTTEND_expt = TOTTTEND_expt.mean(axis=1)


def plot_bar(month):
    plt.figure()
    plt.title('Tendency anomalies')
    plt.bar(np.arange(6), [(hor_diff_expt-hor_diff_cont)[:month].mean(),
                            (vert_diff_expt-vert_diff_cont)[:month].mean(),
                            (hor_adv_expt-hor_adv_cont)[:month].mean(),
                            (vert_adv_expt-vert_adv_cont)[:month].mean(),
                            (diff_expt-diff_cont + hor_adv_expt-hor_adv_cont +
                             vert_adv_expt-vert_adv_cont)[:month].mean(),
                            (TOTTTEND_expt-TOTTTEND_cont)[:month].mean()],
            tick_label=['h. diff', 'v. diff',
                        'h. adv', 'v. adv', 
                        'sum', 'total tendency'],
           align='center')
    plt.hlines(0,-0.5,5.5)
    plt.show()
