#----------------------Define Functions---------------------------
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
    #effective_ice = grounded_ice+ice_below_flotation
    effective_ice_v = effective_ice*grid_area
    esl = np.sum(effective_ice_v)/ocean_area
    return esl
def cal_all_bsl(modern_topo,all_rsl_pred,all_ice):
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
    x_syn = torch.tensor(np.swapaxes(norm_syn_ice,0,1))[None,None,:,:].cuda().float()
    x_syn[x_syn>15] = 15
    x_syn[x_syn<-15] = -15 

    #emulate rsl for both ice history and transfer them back to numpy array
    st.session_state.rsl_syn = model(x_syn).cpu().detach().numpy() 
    #transfer normlized prediction back to original field
    st.session_state.rsl_syn_pred = st.session_state.rsl_syn[0,0]*st.session_state.heal16_output_std[:,None]+st.session_state.heal16_output_mean[:,None]
    st.write('Emulation Done!')
def plot_rsl_comparison():
    '''
    This function is used to generate a relastive sea level comparison plot (i.e., RSL generated by 
    mean ice history and by ice history based on pre-defined ice history
    
    First row: mean ice history, ice history based on pre-defined parameters, and difference between two ice history
    Second row: Emulated RSL based on mean ice history and ice history based on pre-defined parameters and RSL difference
    Third row: Eustatic sea level, emulated RSL at Barbados and Tahiti (two sites show in Fig no. 4) 
    '''
    time_index = 25 - st.session_state.plot_time    
    figure1 =  plt.figure()
    #calculate color scale for plotting 
    max_ice_thick = 100*(np.max([st.session_state.syn_ice[time_index],st.session_state.mean_ice_his[time_index]])//100+1)
    hp.mollview(st.session_state.mean_ice_his[time_index],cmap='turbo',max=max_ice_thick,min=0,hold=True,title='Mean Ice History')
    figure2 =  plt.figure()
    hp.mollview(st.session_state.syn_ice[time_index],cmap='turbo',max=max_ice_thick,min=0,hold=True,title='Synthetic Ice History')
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
   # plt.vlines(25-time_index,0,-120,color='k',linewidth=3)
    plt.legend()
    figure8 =  plt.figure()
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_mean_pred[1130,:],label='Mean ice Barbados RSL',linewidth=3)
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_syn_pred[1130,:],label = 'Synthetic ice Barbados RSL',linewidth=3);
    plt.legend()
    figure9 =  plt.figure()
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_mean_pred[2042,:],label='Mean ice Tahiti RSL',linewidth=3)
    plt.plot(np.arange(25,-1,-1),st.session_state.rsl_syn_pred[2042,:],label = 'Synthetic ice Tahiti RSL',linewidth=3);
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
