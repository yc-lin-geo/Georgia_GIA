import numpy as np
import healpy as hp

#----------------------Define Functions---------------------------
def find_95and68(matrix):
    '''This function is used to find 95% and 68% confidence 
    interval from an ensemble of data
    
    Input:
    matrix: a matrix with the first dimension being the different samples 
    
    '''
    ninetyfive = []
    sixtyeight = []
    for i in range(matrix.shape[1]):
        ninetyfive.append(np.percentile(matrix[:,i],[2.5,97.5]))
        sixtyeight.append(np.percentile(matrix[:,i],[16,84]))
    return np.array(ninetyfive),np.array(sixtyeight)

def cal_bsl(topo,ice):
    '''This function is used to calculate barystatic sea level, which is a direct indicator for 
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

def cal_all_bsl(modern_topo,modern_ice,all_rsl_pred,all_ice):
    '''This function is used to calculate barystatic sea level change history based on 
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
        all_esl[i] = cal_bsl(paleo_topo[:,i],all_ice[i]) 

    all_esl-=all_esl[-1]
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


def cal_ies(ice):
    '''This function is used to calculate ice equivalent sea level.
    
    Input:
    -------------------------
    ice: ice sheet reconstruction at specific time slice, represented as ice heihgt difference (m)
    relative to present-day, in healpix 16 grid
    
    Output:
    -------------------------
    ies: ice equivalent sea level at certain time interval 
    '''
    pho_ice = 893
    pho_water = 1027
    ocean_percent = 0.734375 #assume modern ocean percent is constant
    ocean_area = 4 * np.pi*6371.007**2 * ocean_percent
    grid_area =4 * np.pi*6371.007**2/hp.nside2npix(16)
    
    ice_v =  ice*grid_area
    ies = np.sum(ice_v)/ocean_area
    return ies

def cal_all_ise(all_ice):
    '''This function is used to calculate ice equivalent sea level change history 
    
    Input:
    -------------------------

    all_ice: all ice history (3076 x n)
    
    Output:
    -------------------------
    all_ise: ice equivalent sea level change history at all time slices
    
    '''


    all_ise = np.zeros(all_ice.shape[0])
    for i in range(all_ice.shape[0]):
        all_ise[i] = cal_ies(all_ice[i])
    all_ise[:] -= all_ise[-1]
    return all_ise

def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])
def gen_colormap():
    '''A function to generate a new colormap for plotting Supplementary Figure 1c'''

    from matplotlib import colors
    from matplotlib import cm
        

    cdict = {
    'red':((0.0,inter_from_256(64),inter_from_256(64)),
           (1/5*1,inter_from_256(112),inter_from_256(112)),
           (1/5*2,inter_from_256(230),inter_from_256(230)),
           (1/5*3,inter_from_256(253),inter_from_256(253)),
           (1/5*4,inter_from_256(244),inter_from_256(244)),
           (1.0,inter_from_256(169),inter_from_256(169))),
    'green': ((0.0, inter_from_256(57), inter_from_256(57)),
            (1 / 5 * 1, inter_from_256(198), inter_from_256(198)),
            (1 / 5 * 2, inter_from_256(241), inter_from_256(241)),
            (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
            (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
            (1.0, inter_from_256(23), inter_from_256(23))),
    
    'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
              (1 / 5 * 1, inter_from_256(162), inter_from_256(162)),
              (1 / 5 * 2, inter_from_256(246), inter_from_256(146)),
              (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
              (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
              (1.0, inter_from_256(69), inter_from_256(69))),
    }

    new_cmap = colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)
    return new_cmap

