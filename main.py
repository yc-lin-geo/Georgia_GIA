import streamlit as st
import numpy as np
from scipy import interpolate
import healpy as hp
from matplotlib import pyplot as plt
import os
import torch
#---------------------Seting page layout--------------------------
st.set_page_config(layout="wide",page_title="Deep_GIA")
st.title("A deep-learning based glacial isostatic adjustment emulator")
#st.sidebar.markdown("# Main page 🎈")
st.markdown("""
            Welcome to GEORGIA: a Graphic neural network based EmulatOR for Glacial Isostatic Adjustment! This web-based app
            
            """)
#----------------------Define Functions---------------------------
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

def cal_esl(topo,ice):
    '''This function is used to calculate eustatic sea level based on the theory from 
    Goelzer et al., 2020 The Cryosphere.
    
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
    
    
    ice_f = ice>=1e-3
    ocean_f = (ice + (topo*(pho_water/pho_ice)))<0  #whether ice is grounded
    ocean_f = ~ocean_f*ice_f  #whether ice is grounded
    ocean_f2 = ice_f*topo<0 #whether bedrock is situated below sea level
    
    grounded_ice =  ice*ocean_f
    ice_below_flotation = topo*(pho_water/pho_ice)*ocean_f*ocean_f2 
    effective_ice = grounded_ice+ice_below_flotation
    effective_ice_v = effective_ice*grid_area
    esl = np.sum(effective_ice_v)/ocean_area
    return esl
def cal_all_esl(modern_topo,all_rsl_pred,all_ice):
    '''This function is used to calculate eustatic sea level change history based on 
    reconstructed paleo topography from paleo relative sea level prediction and ice history
    
    Input:
    -------------------------
    modern_topo: modern topography value, written in healpix 16 grid (3076)
    all_rsl_pred: all predicted relative sea level history (3076 x n)
    all_ice: all ice history (3076 x n)
    
    Output:
    -------------------------
    all_esl: eustatic sea level change history at all time slices
    
    '''
    
    all_rsl_pred[:,-1]=0 # make sure modern relative sea level is 0 everywhere
    paleo_topo = modern_topo[:,None] - all_rsl_pred #reconstruct paleo topography
    all_esl = np.zeros(all_rsl_pred.shape[1])
    for i in range(all_rsl_pred.shape[1]):
        all_esl[i] = cal_esl(paleo_topo[:,i],all_ice[i])
    all_esl[:] -= all_esl[-1]
    return all_esl

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
def emulate_GIA():
    '''
    This function is used to emulate GIA based on a ice history
    defined by pre-defined parameters
    '''
    #create random ice history
    random_ice = create_random_combination(st.session_state.healpix16_NA_matrices,[st.session_state.NA_1,st.session_state.NA_2,
                 st.session_state.NA_w_1,st.session_state.NA_w_2,st.session_state.NA_r_1,st.session_state.NA_r_2])
    st.session_state.syn_ice = st.session_state.mean_ice_his.copy()
    st.session_state.syn_ice[:,st.session_state.healpix16_NA_index] = random_ice[:,st.session_state.healpix16_NA_index]
    #the final ice history is the averge of mean ice history and random ice history
    st.session_state.syn_ice[:,st.session_state.healpix16_NA_index] = st.session_state.syn_ice[:,st.session_state.healpix16_NA_index]*0.5+st.session_state.mean_ice_his[:,st.session_state.healpix16_NA_index]*0.5
    
    norm_syn_ice = (st.session_state.syn_ice-st.session_state.heal16_input_mean[None,:])/st.session_state.heal16_input_std[None,:]
    norm_syn_ice[-1,:] = st.session_state.modern_topo_norm
    
    #transfer numpy array to tensors
    x_syn = torch.tensor(np.swapaxes(norm_syn_ice,0,1))[None,None,:,:].cpu().float()
    x_syn[x_syn>15] = 15
    x_syn[x_syn<-15] = -15 

    #emulate rsl for both ice history and transfer them back to numpy array
    st.session_state.rsl_syn = model(x_syn).detach().numpy() 
    #transfer normlized prediction back to original field
    st.session_state.rsl_syn_pred = st.session_state.rsl_syn[0,0]*st.session_state.heal16_output_std[:,None]+st.session_state.heal16_output_mean[:,None]
    st.write('## Emulation complete! Click plotting for visulization!')       

def plot_rsl_comparison():
    '''
    This function is used to generate a relastive sea level comparison plot (i.e., RSL generated by 
    mean ice history and by ice history based on pre-defined ice history
    
    First row: mean ice history, ice history based on pre-defined parameters, and difference between two ice history
    Second row: Emulated RSL based on mean ice history and ice history based on pre-defined parameters and RSL difference
    Third row: Eustatic sea level, emulated RSL at Barbados and Tahiti (two sites show in Fig no. 4) 
    '''
    time_index = 25 - st.session_state.rsl_plot_time    
    figure1 =  plt.figure()
    #calculate color scale for plotting 
    #max_ice_thick = 100*(np.max([st.session_state.syn_ice[time_index],st.session_state.mean_ice_his[time_index]])//100+1)
    max_ice_thick = 3500
    hp.mollview(st.session_state.mean_ice_his[time_index],max=max_ice_thick,min=0,hold=True,title='Mean Ice History')
    figure2 =  plt.figure()
    hp.mollview(st.session_state.syn_ice[time_index],max=max_ice_thick,min=0,hold=True,title='Synthetic Ice History')
    figure3 =  plt.figure()
    max_ice_thick2 = 100*(np.max(np.abs([st.session_state.syn_ice[time_index]-st.session_state.mean_ice_his[time_index]]))//100+1)
    hp.mollview(st.session_state.mean_ice_his[time_index]-st.session_state.syn_ice[time_index],cmap='coolwarm',max=max_ice_thick2,min=-max_ice_thick2,hold=True,title='Mean - Random')
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
    st.session_state.mean_esl = cal_all_esl(st.session_state.modern_topo,st.session_state.rsl_mean_pred,st.session_state.mean_ice_his+st.session_state.modern_ice[None,:])
    st.session_state.syn_esl = cal_all_esl(st.session_state.modern_topo,st.session_state.rsl_syn_pred,st.session_state.syn_ice+st.session_state.modern_ice[None,:])

    plt.plot(np.arange(25,-1,-1),-st.session_state.mean_esl,label='Mean ice ESL',linewidth=3)
    plt.plot(np.arange(25,-1,-1),-st.session_state.syn_esl,label = 'Synthetic ice ESL',linewidth=3)
    plt.vlines(st.session_state.rsl_plot_time,0,-130,color='k',label='Plotting Time',linewidth=3,linestyle = ':')
    plt.legend()
    figure8 =  plt.figure()
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_mean_pred[1130,:],label='Mean ice Barbados RSL',linewidth=3)
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_syn_pred[1130,:],label = 'Synthetic ice Barbados RSL',linewidth=3);
    plt.vlines(st.session_state.rsl_plot_time,0,-130,color='k',label='Plotting Time',linewidth=3,linestyle = ':')
    plt.legend()
    figure9 =  plt.figure()
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_mean_pred[2042,:],label='Mean ice Tahiti RSL',linewidth=3)
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_syn_pred[2042,:],label = 'Synthetic ice Tahiti RSL',linewidth=3);
    plt.vlines(st.session_state.rsl_plot_time,0,-130,color='k',label='Plotting Time',linewidth=3,linestyle = ':')
    plt.legend()
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

#-----------------------Load Inputs-------------------------------

#Load mean and standard deviation for inputs and outputs for normlization
st.session_state.heal16_input_mean = np.load('./data/heal16_input_mean.npy')
st.session_state.heal16_input_std = np.load('./data/heal16_input_std.npy')
st.session_state.heal16_output_mean = np.load('./data/heal16_output_mean.npy')
st.session_state.heal16_output_std = np.load('./data/heal16_output_std.npy') 
#Load mean ice history based on all training dataset and its corresponding RSL history for comparison
st.session_state.mean_ice_his = np.load('./data/heal16_input_mean_his.npy')
st.session_state.rsl_mean_pred = np.load('./data/healpix16_mean_ice_rsl.npy')

#Load modern ice thickness and topography
st.session_state.modern_topo = np.load('./data/modern_topo_healpix16.npy')
st.session_state.modern_topo_norm = np.load('./data/modern_topo_norm.npy')
st.session_state.modern_ice = np.load('./data/ice_0_healpix16.npy')
#Load ice histories from previous studies
st.session_state.healpix16_NA_matrices = np.load('./data/healpix16_NA_matrcies.npy')
st.session_state.healpix16_NA_index = np.sum(st.session_state.healpix16_NA_matrices,axis=0)[5]!=0
#load emulator
model = torch.load('./emulator/GIA_emulator').to('cpu')

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
                step=3,
                key='NA_r_2') 

st.sidebar.slider("Ice history plotting time (1-24 ka BP)",
                min_value = 1,
                max_value = 24,
                step=1,
                key='plot_time')

#-----------------------Plotting different ice models-----------------------
st.write('## North American ice history at '+str(st.session_state.plot_time)+' ka BP')
#calculate synthetic ice volume 
synthetic_ice = create_random_combination(st.session_state.healpix16_NA_matrices,[st.session_state.NA_1,st.session_state.NA_2,
                 st.session_state.NA_w_1,st.session_state.NA_w_2,st.session_state.NA_r_1,st.session_state.NA_r_2])
synthetic_ice_v =  np.sum(cal_ice_v(synthetic_ice),axis=1)
mean_NA = np.mean(st.session_state.healpix16_NA_matrices,axis=0)
mean_ice_v = np.sum(cal_ice_v(mean_NA),axis=1)

#Generate figures here
all_ice_v = plt.figure()
plot_index = 25-st.session_state.plot_time
for i in range(len(st.session_state.healpix16_NA_matrices)):
    test_ice_v = np.sum(cal_ice_v(st.session_state.healpix16_NA_matrices[i]),axis=1)
    plt.plot(np.arange(25,-1,-1),test_ice_v,label='ice_history_'+str(i))
plt.plot(np.arange(25,-1,-1),mean_ice_v,color='grey',linestyle='--',linewidth=3,label='Mean ice history')
plt.plot(np.arange(25,-1,-1),synthetic_ice_v,color='k',linewidth=3,linestyle='--',label='Synthetic ice history')
plt.vlines(st.session_state.plot_time,0,4,color='steelblue',label='Plotting Time',linewidth=3,linestyle = ':')
plt.xlabel('Time (ka BP)',fontsize=10)
plt.ylabel('Ice volume (10$^9$ m$^3$)',fontsize=10)
plt.legend() 
plt.title('Ice volume history',fontsize=10) 
 
 
#ice_thick_max = 100*(np.max(st.session_state.healpix16_NA_matrices[:,plot_index])//100+1)
ice_thick_max = 3500
#Mean ice history
Mean_ice_his = plt.figure()
hp.mollview(mean_NA[plot_index],hold=True,max=ice_thick_max,min=0,title='Synthetic ice history')
#Synthetic ice history
Syn_ice_his = plt.figure()
hp.mollview(synthetic_ice[plot_index],hold=True,max=ice_thick_max,min=0,title='Synthetic ice history')

#All ice histories from previous studies
NA1 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[0,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 0')

NA2 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[1,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 1')

NA3 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[2,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 2')

NA4 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[3,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 3')

NA5 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[4,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 4')

NA6 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[5,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 5')

NA7 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[6,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 6')

NA8 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[7,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 7')

NA9 = plt.figure()
hp.mollview(st.session_state.healpix16_NA_matrices[8,plot_index],hold=True,max=ice_thick_max,min=0,title='Ice history 8')

#Arange figure for plotting
ice_his0 = st.container()
coll0, coll1,coll11 = st.columns(3)
with ice_his0:
    with coll0:
        all_ice_v
    with coll1:
        Mean_ice_his
    with coll11:
        Syn_ice_his
    

st.write('## Check all available North American ice histories from previous studies with the expander below!!')
with st.expander('Click here!'):

    ice_his1 = st.container()
    coll2, coll3,coll4 = st.columns(3)

    with ice_his1:
        with coll2:
            NA1
        with coll3:
            NA2
        with coll4:
            NA3

    ice_his2 = st.container()
    coll5, coll6, coll7 = st.columns(3)

    with ice_his2:
        with coll5:
            NA4
        with coll6:
            NA5
        with coll7:
            NA6

    ice_his3 = st.container()
    coll8, coll9, coll10 = st.columns(3)

    with ice_his3:
        with coll8:
            NA7
        with coll9:
            NA8
        with coll10:
            NA9

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
    


