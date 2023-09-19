import streamlit as st
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import torch
import pandas as pd
import matplotlib

font = {'weight' : 'normal',
        'size'   : 20}
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['legend.frameon'] = 'False'
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rc('font',**font)

#------------------Try to import a subroutine----------------------------
import subprocess
import sys
import time
try:
  # replace "yourpackage" with the package you want to import
  import pygsp
except ModuleNotFoundError as e:
    # This block executes only on the first run when your package isn't installed
    subprocess.Popen([f'{sys.executable} -m pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP'], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(90)

#---------------------Seting page layout--------------------------
st.set_page_config(layout="wide",page_title="GEORGIA")
st.title("GEORGIA: a Graphic neural network based EmulatOR for Glacial Isostatic Adjustment")
st.markdown("###### by Yucheng Lin - yc.lin@rutgers.edu ")
#st.sidebar.markdown("# Main page 🎈")
st.markdown("""
            Welcome to GEORGIA: a Graphic neural network based EmulatOR for Glacial Isostatic Adjustment! This web-based API is designed to explore different North American Ice Sheet (NAIS) histories impact on global relative sea level. 
           
            See more details in [paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL103672) and [Github](https://github.com/yc-lin-geo/Georgia_GIA/tree/master).
            """)

# Expander Styling

st.markdown(
    """
<style>
.streamlit-expanderHeader {
 #   font-weight: bold;
    background: aliceblue;
    font-size: 18px;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.expander("See more information about this GEORGIA API"):
    st.write("""You can generate your own synthetic NAIS model by randomly combine 2 previously published NAIS models from: 0-ICE5G (Peltier et al., 2002), 1-ANU(Lambeck et al., 2017), 2-ICE6G_C (Peltier et al., 2015), 3-ICE7G_NA (Roy et al., 2017), 4-GLAC_1D 9894 (Tarasov et al., 2012), 5-GLAC1D_9927 (Tarasov et al., 2012), 6-PaleoMIST (Gowan et al., 2021), 7-NAICE (Gowan et al., 2017) and 8-Han_2021 (Han et al., 2021). You can see all available ice geometry by clicking 'Show all available ice geometry button on the bottom of right panel'. 
 
            You can use the slider bar on the left hand side to change the paramters to generate your synthetic ice history, the resulting ice history will be shown on the right panel:
            - NAIS index 1 and 2: Use these two parameters to select two listed NAIS model above to combine a new NAIS model!
            - NAIS weighting factor 1 and 2: Use these two parameters to control relative weighting between the two selected ice model. Note, these two are relative weighting factor, i.e., 1 and 1 will produce the same ice history as 0.5 and 0.5, etc. 
            - NAIS temporal factor 1 and 2: Use these two parameters to make the selected ice history deglaciate earlier (negative value) or later (positive value). 
            - Ice history plotting time: Use this parameter to control which time slice you want to visualize. 
           
Once you satisfy with the current synthetic ice history, you can click 'Run emulation' button on the bottom of left bar, which will automatically emulate the corresponding global relative sea-level change history by combining your synthetic ice history one specific Earth rheological model. You can then use the slider to change the time slice of relative sea level and hit 'Plot emulation results!' for a visulization. Lastly, you can download the synthetic ice histories along with the emulated RSL result by the 'Download Results!' button below. Please note that all of the RSL predictions assume one specific Earth model with 71 km lithosphere, 0.3/70 x 10**21 Pa s upper/lower mantle viscosity.

 """)

#----------------------Define Functions---------------------------
#Button style
customized_button = st.markdown("""
    <style >
    .stDownloadButton, div.stButton {text-align:center}
    .stDownloadButton button, div.stButton > button:first-child {
        background-color: #77b5c9;
        color:#000000;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 6px;
        border-width: 3px;
        border-image: initial;
        font-size: 18px;
    }
    
    .stDownloadButton button:hover, div.stButton > button:hover {
        background-color: #77b5c9;
        color:#000000;
    }
        }
    </style>""", unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def cal_ice_v(ice):
    '''This function is used to calculate ice volume
    
    Input:
    -------------------------
    ice: ice sheet reconstruction at specific time slice, represented as ice heihgt difference (m)
    relative to present-day, in healpix 16 grid
    
    Output:
    -------------------------
    ies: ice volume in m^3
    '''
    grid_area =4 * np.pi*6371.007**2/hp.nside2npix(16)
    
    ice_v =  ice*grid_area
    return ice_v/1e10

@st.cache(suppress_st_warning=True)
def ice_roll(ice_matrix,slide_num):
    '''This function is used to genrate ice history with temporal rolling
    
    Iuputs:
    --------------------------
    ice_matrix: 2d numpy array, containing temporal x spatial ice history
    slide_num: an integer, containing the rolling number 
    
    Output:
    -------------------------
    rolled_ice: rolled ice matrix
    '''
    
    rolled_ice = np.roll(ice_matrix, slide_num,axis=0)
    
    if slide_num>=0:
        rolled_ice[:slide_num] = rolled_ice[slide_num] 
    else:
        rolled_ice[slide_num:] = rolled_ice[slide_num-1] 
    return rolled_ice 

@st.cache(suppress_st_warning=True)
def cal_esl2(topo,ice):
    '''This function is used to calculate eustatic sea level, which is a direct indicator for 
    grounded ice volume. In this case, only the floating ice is excluded from calculating ESL. 
    
    Inputs:
    -------------------------
    topo: topography reconstruction at specific time slice, represented in healpix 16 grid
    ice: ice sheet reconstruction at specific time slice, represented as ice heihgt difference (m)
    relative to present-day, in healpix 16 grid
    
    Output:
    esl: eustatic sea level at certain time interval'''
    pho_ice = 893
    pho_water = 1027
    ocean_percent = np.mean(topo<0)
    ocean_area = 4 * np.pi*6371.007**2 * ocean_percent
    grid_area =4 * np.pi*6371.007**2/hp.nside2npix(16)
    
    
    ice_f = ice>=1e-3 #define the ice area
    ocean_f = (ice + (topo*(pho_water/pho_ice)))<0  #whether ice is grounded
    ocean_f = ~ocean_f*ice_f  #whether ice is grounded
    
    grounded_ice =  ice*ocean_f
    grounded_ice_v = grounded_ice*grid_area
    esl = np.sum(grounded_ice_v)/ocean_area
    return esl

@st.cache(suppress_st_warning=True)
def cal_all_esl2(modern_topo,modern_ice,all_rsl_pred,all_ice):
    '''This function is used to calculate eustatic sea level change history based on 
    reconstructed paleo topography from paleo relative sea level prediction and ice history
    
    Input:
    -------------------------
    modern_topo: modern topography value, written in healpix 16 grid (3076)
    modern_ice: modern ice thickness, written in healpix 16 grid (3076)
    all_rsl_pred: all predicted relative sea level history (3076 x n)
    all_ice: all ice history (3076 x n)
    
    Output:
    -------------------------
    all_esl: eustatic sea level change history at all time slices
    
    '''
    
    all_rsl_pred[:,-1]=0 # make sure modern relative sea level is 0 everywhere
    paleo_topo = modern_topo[:,None] - all_rsl_pred #reconstruct paleo topography
    all_esl = np.zeros(all_rsl_pred.shape[1])
    all_ice = modern_ice+all_ice
    for i in range(all_rsl_pred.shape[1]):
        all_esl[i] = cal_esl2(paleo_topo[:,i],all_ice[i]) 

    all_esl-=all_esl[-1]
    return all_esl


@st.cache(suppress_st_warning=True)
def create_random_combination(ice_matrices,random_factors):
    '''
    This function is used to create random combination of ice models based on provided random factors
    
    Inputs:
    -----------------------------------
    ice_matrices: 3d matrix containing n1 ice models, n2 ice sptatial grids, n3 temporal grids
    random_factors: [model_index_1,model_index_2,w_1,w_2,r_1,r_2]
    where model index corresponds to the existing model index, w_factor corresponds to random weighting 
    factor and r_factor corresponds to random rolling factor
    Outputs:
    -----------------------------------
    random_ice: random ice model based on random ice factor
    '''
    
    random_index = random_factors[:2]
    random_weights = random_factors[2:4]
    roll_index = random_factors[4:6]
    ice_copy = ice_matrices.copy()
    for i in range(2):
        ice_copy[random_index[i]] = ice_roll(ice_copy[random_index[i]],roll_index[i])
    random_ice = np.average(ice_copy[random_index],weights = random_weights,axis=0)
    
    return random_ice

@st.cache(suppress_st_warning=True)
def emulate_GIA():
    '''
    This function is used to emulate GIA based on a ice history
    defined by pre-defined parameters
    '''
    #create random ice history
    #random_ice = create_random_combination(st.session_state.healpix16_NA_matrices,[st.session_state.NA_1,st.session_state.NA_2,
#                 st.session_state.NA_w_1,st.session_state.NA_w_2,st.session_state.NA_r_1,st.session_state.NA_r_2])
    syn_ice = st.session_state.mean_ice_his.copy()
    syn_ice[:,st.session_state.healpix16_NA_index] = st.session_state.synthetic_ice[:,st.session_state.healpix16_NA_index]
    #the final ice history is the averge of mean ice history and random ice history
    syn_ice[:,st.session_state.healpix16_NA_index] = syn_ice[:,st.session_state.healpix16_NA_index]*0.5+st.session_state.mean_ice_his[:,st.session_state.healpix16_NA_index]*0.5
    
    norm_syn_ice = (syn_ice-st.session_state.heal16_input_mean[None,:])/st.session_state.heal16_input_std[None,:]
    norm_syn_ice[-1,:] = st.session_state.modern_topo_norm
    
    #transfer numpy array to tensors
    x_syn = torch.tensor(np.swapaxes(norm_syn_ice,0,1))[None,None,:,:].cpu().float()
    x_syn[x_syn>15] = 15
    x_syn[x_syn<-15] = -15 
    
    #emulate rsl for both ice history and transfer them back to numpy array
    rsl_syn = model(x_syn).detach().numpy() 
    #transfer normlized prediction back to original field
    st.session_state.rsl_syn_pred = rsl_syn[0,0]*st.session_state.heal16_output_std[:,None]+st.session_state.heal16_output_mean[:,None] 
    st.write('### Emulation complete! Click plotting for visulization!', unsafe_allow_html=True)   
    st.session_state.georgia_results = create_results()
    del syn_ice,norm_syn_ice,x_syn,rsl_syn

@st.cache(suppress_st_warning=True)
def plot_rsl_comparison():
    '''
    This function is used to generate a relastive sea level comparison plot (i.e., RSL generated by 
    mean ice history and by ice history based on pre-defined ice history
    
    First row: mean ice history, ice history based on pre-defined parameters, and difference between two ice history
    Second row: Emulated RSL based on mean ice history and ice history based on pre-defined parameters and RSL difference
    Third row: Eustatic sea level, emulated RSL at Barbados and Tahiti (two sites show in Fig no. 4) 
    '''
    font = {'weight' : 'normal',
        'size'   : 20}
    matplotlib.rcParams['xtick.major.size'] = 8
    matplotlib.rcParams['ytick.major.size'] = 8
    matplotlib.rcParams['axes.labelsize'] = 22
    matplotlib.rcParams['legend.frameon'] = 'False'
    matplotlib.rcParams['figure.figsize'] = (10,6)
    matplotlib.rc('font',**font)
    st.session_state.emu_stage = 1
    time_index = 25 - st.session_state.rsl_plot_time    
    syn_ice = st.session_state.mean_ice_his.copy()
    syn_ice[:,st.session_state.healpix16_NA_index] = st.session_state.synthetic_ice[:,st.session_state.healpix16_NA_index]
    syn_ice[:,st.session_state.healpix16_NA_index] = syn_ice[:,st.session_state.healpix16_NA_index]*0.5+st.session_state.mean_ice_his[:,st.session_state.healpix16_NA_index]*0.5

    figure1 =  plt.figure()
    #calculate color scale for plotting 
    #max_ice_thick = 100*(np.max([st.session_state.syn_ice[time_index],st.session_state.mean_ice_his[time_index]])//100+1)
    max_ice_thick = 3500
    hp.mollview(st.session_state.mean_ice_his[time_index],max=max_ice_thick,min=0,hold=True,title='Mean Ice History')
    figure2 =  plt.figure()
    hp.mollview(syn_ice[time_index],max=max_ice_thick,min=0,hold=True,title='Synthetic Ice History')
    figure3 =  plt.figure()
    max_ice_thick2 = 100*(np.max(np.abs([syn_ice[time_index]-st.session_state.mean_ice_his[time_index]]))//100+1)
    hp.mollview(st.session_state.mean_ice_his[time_index]-syn_ice[time_index],cmap='coolwarm',max=max_ice_thick2,min=-max_ice_thick2,hold=True,title='Mean ice - Synthetic ice')
    figure4 =  plt.figure()
    #calculate color scale for plotting
    max_rsl = 10*(-np.max([st.session_state.rsl_mean_pred[2042,time_index],st.session_state.rsl_syn_pred[2042,time_index]])//10+1)
    hp.mollview(st.session_state.rsl_mean_pred[:,time_index],cmap='RdBu_r',max=max_rsl,min=-max_rsl,hold=True,title='Mean RSL')
    hp.visufunc.projscatter([-300.9375,-210.42],[14.477512185929921,-17.553],color='r',lonlat=True,marker='^',s=100,
                       edgecolors='k')
    figure5 =  plt.figure()
    hp.mollview(st.session_state.rsl_syn_pred[:,time_index],cmap='RdBu_r',max=max_rsl,min=-max_rsl,hold=True,title='Synthetic RSL')
    figure6 =  plt.figure()
    max_rsl_dif = 2*(np.max(np.abs([st.session_state.rsl_mean_pred[2042,time_index]-st.session_state.rsl_syn_pred[2042,time_index]]))//2+1)
    hp.mollview(hp.sphtfunc.smoothing(st.session_state.rsl_mean_pred[:,time_index]-st.session_state.rsl_syn_pred[:,time_index],sigma=0.05),cmap='coolwarm',max=max_rsl_dif,min=-max_rsl_dif,hold=True,title='Mean RSL - Synthetic RSL')
    figure7 =  plt.figure()
    mean_esl = cal_all_esl2(st.session_state.modern_topo,st.session_state.modern_ice,st.session_state.rsl_mean_pred,st.session_state.mean_ice_his)
    syn_esl = cal_all_esl2(st.session_state.modern_topo,st.session_state.modern_ice,st.session_state.rsl_syn_pred,syn_ice)
    plt.title('Eustatic sea level history')
    plt.plot(np.arange(25,-1,-1),-mean_esl,label='Mean ice ESL',linewidth=5)
    plt.plot(np.arange(25,-1,-1),-syn_esl,label = 'Synthetic ice ESL',linewidth=5)
    plt.vlines(st.session_state.rsl_plot_time,0,-130,color='k',label='Plotting Time',linewidth=5,linestyle = '--')
    plt.legend(loc=3)
    figure8 =  plt.figure()
    plt.title('Relative sea level history at Barbados')
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_mean_pred[1130,:],label='Mean ice Barbados RSL',linewidth=5)
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_syn_pred[1130,:],label = 'Synthetic ice Barbados RSL',linewidth=5);
    plt.vlines(st.session_state.rsl_plot_time,0,-130,color='k',label='Plotting Time',linewidth=5,linestyle = '--')
    plt.legend(loc=3)
    figure9 =  plt.figure()
    plt.title('Relative sea level history at Tahiti')
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_mean_pred[2042,:],label='Mean ice Tahiti RSL',linewidth=5)
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_syn_pred[2042,:],label = 'Synthetic ice Tahiti RSL',linewidth=5);
    plt.vlines(st.session_state.rsl_plot_time,0,-130,color='k',label='Plotting Time',linewidth=5,linestyle = '--')
    plt.legend(loc=3)
    plt.tight_layout()
    #st.pyplot(fig) 
    container1 = st.container() 
    col1, col2, col3 = st.columns(3) 
    with container1:
        with col1:
            figure1
        with col2:
            figure2
        with col3: 
            figure3
   
    container2 = st.container()
    col4, col5, col6 = st.columns(3)

    with container2:
        with col4:
            figure4
        with col5:
            figure5
        with col6:
            figure6

    container3 = st.container()
    col7, col8, col9 = st.columns(3)

    with container3:
        with col7:
            figure7
        with col8:
            figure8
        with col9:
            figure9
    st.download_button(label='Download Results!', data=st.session_state.georgia_results,file_name='GEORGIA_results.csv',mime='text/csv')
    del syn_ice
@st.cache
def create_results():
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    header = ['Lon','Lat']+['Ice_' +str(i)+'_ka' for i in range(25,0,-1)]+['RSL_' +str(i)+'_ka' for i in range(25,0,-1)]
    df = pd.concat([pd.DataFrame(st.session_state.healpix16_coords),pd.DataFrame(np.swapaxes(st.session_state.synthetic_ice,0,1)[:,:-1]),pd.DataFrame(st.session_state.rsl_syn_pred[:,:-1])], axis=1,ignore_index=True)
    df.columns = header
    
    return df.to_csv().encode('utf-8')
     
#-----------------------Load Inputs-------------------------------
@st.cache
def load_model(max_entries=1):
        
	#load emulator
	model = torch.load('./emulator/GIA_emulator').to('cpu')
	return model

model = load_model()

@st.cache
def load_data(max_entries=13):
        
	#Load mean and standard deviation for inputs and outputs for normlization
	heal16_input_mean = np.load('./data/heal16_input_mean.npy',allow_pickle=True)
	heal16_input_std = np.load('./data/heal16_input_std.npy',allow_pickle=True)
	heal16_output_mean = np.load('./data/heal16_output_mean.npy',allow_pickle=True)
	heal16_output_std = np.load('./data/heal16_output_std.npy',allow_pickle=True) 
	#Load mean ice history based on all training dataset and its corresponding RSL history for comparison
	mean_ice_his = np.load('./data/heal16_input_mean_his.npy',allow_pickle=True)
	rsl_mean_pred = np.load('./data/healpix16_mean_ice_rsl.npy',allow_pickle=True)

	#Load modern ice thickness and topography
	modern_topo = np.load('./data/modern_topo_healpix16.npy',allow_pickle=True)
	modern_topo_norm = np.load('./data/modern_topo_norm.npy',allow_pickle=True)
	modern_ice = np.load('./data/ice_0_healpix16.npy',allow_pickle=True)
	#Load ice histories from previous studies
	healpix16_NA_matrices = np.load('./data/healpix16_NA_matrcies.npy',allow_pickle=True)
	healpix16_NA_index = np.sum(healpix16_NA_matrices,axis=0)[5]!=0

	#coordinate of healpix grids
	healpix_16_coords = pd.read_csv('./data/healpix16_coord.csv',header=None)
	return heal16_input_mean,heal16_input_std,heal16_output_mean,heal16_output_std,mean_ice_his,rsl_mean_pred,modern_topo,modern_topo_norm,modern_ice,healpix16_NA_matrices,healpix16_NA_index,healpix_16_coords

st.session_state.heal16_input_mean,st.session_state.heal16_input_std,st.session_state.heal16_output_mean,st.session_state.heal16_output_std,st.session_state.mean_ice_his,st.session_state.rsl_mean_pred,st.session_state.modern_topo,st.session_state.modern_topo_norm,st.session_state.modern_ice,st.session_state.healpix16_NA_matrices,st.session_state.healpix16_NA_index,st.session_state.healpix16_coords = load_data() 
#---------------------Set ice model parameters--------------------
# Sessions tate initialise
# Check if 'key' already exists in session_state
# If not, then initialize it
if 'NA_1' not in st.session_state:
    st.session_state['NA_1'] = 0

if 'NA_2' not in st.session_state:
    st.session_state['NA_2'] = 1

if 'NA_w_1' not in st.session_state:
    st.session_state['NA_w_1'] =0.5

if 'NA_w_2' not in st.session_state:
    st.session_state['NA_w_2'] = 0.5

if 'NA_r_1' not in st.session_state:
    st.session_state['NA_r_1'] =0

if 'NA_r_2' not in st.session_state:
    st.session_state['NA_r_2'] = 0

if 'plot_time' not in st.session_state:
    st.session_state['plot_time'] = 21

if 'rsl_plot_time' not in st.session_state:
    st.session_state['rsl_plot_time'] = 21

if 'rsl_syn_pred' not in st.session_state:
    st.session_state.rsl_syn_pred = np.zeros([3072,26])

if 'emu_stage' not in st.session_state:
    st.session_state.emu_stage = 0

if 'georgia_results' not in st.session_state:
    st.session_state.georgia_results = 0
#-------------------Set side bar for parameter setting-----------------------------
st.sidebar.write('## Generate synthic ice history based on paramters here!')
st.sidebar.slider("NAIS index 1",
                min_value = 0,
                max_value = 8,
                step=1,
                key='NA_1')
st.sidebar.slider("NAIS index2",
                min_value = 0,
                max_value = 8,
                step=1,
                key='NA_2')
    
st.sidebar.slider("NAIS weighting factor 1",
                min_value = 0.,
                max_value = 1., 
                key='NA_w_1')

st.sidebar.slider("NAIS weighting factor 2",
                min_value = 0.,
                max_value = 1.,
                key='NA_w_2')

st.sidebar.slider("NAIS temporal factor 1",
                min_value = -3,
                max_value = 3,
                step=1,
                key='NA_r_1')

st.sidebar.slider("NAIS temporal factor 2",
                min_value = -3,
                max_value = 3,
                step=1,
                key='NA_r_2') 

st.sidebar.slider("Ice history plotting time (1-24 ka BP)",
                min_value = 1,
                max_value = 24,
                step=1,
                key='plot_time')

#-----------------------Plotting different ice models-----------------------

st.write('### North American ice history at '+str(st.session_state.plot_time)+' ka BP')
#calculate synthetic ice volume 
st.session_state.synthetic_ice = create_random_combination(st.session_state.healpix16_NA_matrices,[st.session_state.NA_1,st.session_state.NA_2,st.session_state.NA_w_1,st.session_state.NA_w_2,st.session_state.NA_r_1,st.session_state.NA_r_2])
#synthetic_ice_v =  np.sum(cal_ice_v(st.session_state.synthetic_ice),axis=1)
#mean_NA = np.mean(st.session_state.healpix16_NA_matrices,axis=0)
#mean_ice_v = np.sum(cal_ice_v(mean_NA),axis=1)

#Generate figures here
#all_ice_v = plt.figure()
#plot_index = 25-st.session_state.plot_time
NA_model_name = ['0 ICE5G','1 ANU','2 ICE6G_C','3 ICE7G_NA','4 GLAC1D_9894','5 GLAC1D_9927','6 PaleoMIST','7 NAICE','8 Han_2021']

@st.cache(suppress_st_warning=True)
def plot_syn_ice_comparison():
	synthetic_ice_v =  np.sum(cal_ice_v(st.session_state.synthetic_ice),axis=1)
	mean_NA = np.mean(st.session_state.healpix16_NA_matrices,axis=0)
	mean_ice_v = np.sum(cal_ice_v(mean_NA),axis=1) 
	all_ice_v = plt.figure()
	plot_index = 25-st.session_state.plot_time

	for i in range(len(st.session_state.healpix16_NA_matrices)):
                test_ice_v = np.sum(cal_ice_v(st.session_state.healpix16_NA_matrices[i]),axis=1)
                if (i ==st.session_state.NA_1) or (i==st.session_state.NA_2):
                    plt.plot(np.arange(25,-1,-1),test_ice_v,label=NA_model_name[i],linewidth=3.5,linestyle='--')
                else:
                    plt.plot(np.arange(25,-1,-1),test_ice_v,color='grey',linewidth=0.6,alpha=0.6)
        
	plt.plot(np.arange(25,-1,-1),mean_ice_v,color='purple',linewidth=4.5,label='Mean ice history')
	plt.plot(np.arange(25,-1,-1),synthetic_ice_v,color='k',linewidth=4.5,label='Synthetic ice history')
	plt.vlines(st.session_state.plot_time,0,4,color='pink',label='Plotting Time',linewidth=3.5,linestyle = '--')
	plt.xlabel('Time (ka BP)',fontsize=15)
	plt.ylabel('Ice volume (10$^9$ m$^3$)',fontsize=15)
	plt.legend() 
	plt.title('Ice volume history',fontsize=15) 
 
 
	#ice_thick_max = 100*(np.max(st.session_state.healpix16_NA_matrices[:,plot_index])//100+1)
	ice_thick_max = 3500
	#Mean ice history
	Mean_ice_his = plt.figure()
	hp.mollview(mean_NA[plot_index],hold=True,max=ice_thick_max,min=0,title='Mean ice geometry')
	#Synthetic ice history
	Syn_ice_his = plt.figure()
	hp.mollview(st.session_state.synthetic_ice[plot_index],hold=True,max=ice_thick_max,min=0,title='Synthetic ice geometry')
	#Difference between mean and syn ice

	Dif_ice_his = plt.figure()
	hp.mollview(st.session_state.synthetic_ice[plot_index]-mean_NA[plot_index],hold=True,max=500,min=-500,cmap='coolwarm',title='Synthetic ice - mean ice')

	#Arange figure for plotting
	ice_his0 = st.container()
	coll0, coll1,coll11,coll111 = st.columns(4)
	with ice_his0:
	    with coll0:
	        all_ice_v
	    with coll1:
	        Mean_ice_his
	    with coll11:
	        Syn_ice_his
	    with coll111:
	        Dif_ice_his
	if st.session_state.emu_stage==1:
		st.experimental_memo.clear()
		#st.cache_data.clear()
		st.session_state.emu_stage=0
		
plot_syn_ice_comparison()
expander =  st.expander("Show all available ice geometries at "+str(st.session_state.plot_time)+'ka BP')

@st.cache(suppress_st_warning=True)
def plot_all_models(healpix16_NA_matrices, plot_index):
    ice_thick_max = 3500
    fig, axs = plt.subplots(2,5,figsize=(40, 10))
    plt.subplot(2,5,1)
    hp.mollview(healpix16_NA_matrices[0,plot_index],hold=True,max=ice_thick_max,min=0,title='ICE5G')
    plt.subplot(2,5,2)
    hp.mollview(healpix16_NA_matrices[1,plot_index],hold=True,max=ice_thick_max,min=0,title='ANU')
    plt.subplot(2,5,3)
    hp.mollview(healpix16_NA_matrices[2,plot_index],hold=True,max=ice_thick_max,min=0,title='ICE6G_C')
    plt.subplot(2,5,4)
    hp.mollview(healpix16_NA_matrices[3,plot_index],hold=True,max=ice_thick_max,min=0,title='ICE7G_NA')
    plt.subplot(2,5,5)
    hp.mollview(healpix16_NA_matrices[4,plot_index],hold=True,max=ice_thick_max,min=0,title='GLAC1D_9894')
    plt.subplot(2,5,6)
    hp.mollview(healpix16_NA_matrices[5,plot_index],hold=True,max=ice_thick_max,min=0,title='GLAC1D_9927')
    plt.subplot(2,5,7)
    hp.mollview(healpix16_NA_matrices[6,plot_index],hold=True,max=ice_thick_max,min=0,title='PaleoMIST')
    plt.subplot(2,5,8)
    hp.mollview(healpix16_NA_matrices[7,plot_index],hold=True,max=ice_thick_max,min=0,title='NAICE')
    plt.subplot(2,5,9)
    hp.mollview(healpix16_NA_matrices[8,plot_index],hold=True,max=ice_thick_max,min=0,title='Han_2021')
    ax= plt.subplot(2,5,10)
    ax.axis('off')

    plt.tight_layout(pad=0.001, w_pad=0.0001, h_pad=0.001)

    st.pyplot(fig)

if expander.button("Plot all"):
    
    plot_all_models(st.session_state.healpix16_NA_matrices,25-st.session_state.plot_time)
  
#-----------------------------Start RSL emulation!-----------------------------
with st.sidebar.form(key='my_form'):
    st.subheader('Start emulation based on the ice here above!')
    
    submit_button = st.form_submit_button(label='Run emulation!', on_click=emulate_GIA)
    st.slider("Change plotting time for emulation results (1-24 ka BP)",
                min_value = 1,
                max_value = 24,
                step=1,
                key='rsl_plot_time')
    submit_button = st.form_submit_button(label='Plot emulation results!', on_click=plot_rsl_comparison)
    

