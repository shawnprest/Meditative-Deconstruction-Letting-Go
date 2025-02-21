from pymdp.agent import Agent
from pymdp import utils, maths, control, learning
import numpy as np
from copy import deepcopy

def init_V1_generative_model(T=3, verbose=False, reward_for_stopping=0.001):
    
    # State Factors (s)
    # -----------------

    # SF1 - various visual things that can manifest
    # what_states
    facial_feature_state_names = ['null', 'eye_left', 'eye_right', 'mouth_up', 'mouth_down']

    # SF2 - Proprioceptive choice states for agent
    # where_states
    sampling_state_names = ['sampling' , 'stopping']

    # SF3 - Affective valence states
    # valence_states
    valence_state_names = ['good', 'null']

    # get num of states
    num_states = [len(facial_feature_state_names), len(sampling_state_names), len(valence_state_names)] # 5*2 matrix - the slices will be sampling states
    if verbose: print(f'Gen Process Setup: num_states: {num_states}') # [5, 2] # 5 facial, 2 sampling


    #there are two hidden state factors, facial features & sampling states
    num_factors = len(num_states)
    if verbose: print(f'Gen Process Setup: num_factors (state factors): {num_factors}')

    # Action Control State Factors (u)
    # --------------------------------

    facial_feature_action_names = ['doing_nothing'] # You cannot change what feature appears
    # action_names
    sampling_action_names = ['go_sample', 'stop_sampling'] # You can go and sample or stop sampling

    valence_action_names = ['doing_nothing']
    # the number control state factors (here, only 1)
    # here, one hidden facial feature does not mean a transition to another
    
    # use the following instead since there are no necessary facial feature actions, just doing_nothing and no necessary valence actions, just doing_nothing
    num_controls = [1, len(sampling_action_names), 1] # we transition from 'eye_left' to 'eye_left', etc.
    if verbose: print(f'Gen Process Setup: num_controls: {num_controls}')

    # for every hidden state factor there is a control state factor
    num_control_factors = len(num_controls) # total number of control state factors
    if verbose: print(f'Gen Process Setup: num_control_factors: {num_control_factors}')

    # Observation Modalities (o)
    # --------------------------

    # Observations for SF1 (facial features)
    facial_feature_obs_names = ['null', 'seeing_eye_left', 'seeing_eye_right', 'seeing_mouth_up', 'seeing_mouth_down']

    # Observations for SF2 (sampling proprioceptive feedback)
    sampling_obs_names = ['noticed_sampling', 'noticed_stopping']

    # Observations for SF3 (valence observations)
    valence_obs_names = ['noticing_v1', 'null']

    # get num of observations of features
    num_obs = [len(facial_feature_obs_names), len(sampling_obs_names),len(valence_obs_names)] # this is the number of observation modalities
    if verbose: print(f'Gen Process Setup: num_obs: {num_obs}')

    num_obs_modalities = len(num_obs)
    if verbose: print(f'Gen Process Setup: num_obs_modalities: {num_obs_modalities}')

    dimensions = {  'num_states' : num_states,
                'num_obs' : num_obs,
                'num_controls' : num_controls }

    mappings = { 'facial_feature_state_names' : facial_feature_state_names,
                'sampling_state_names' : sampling_state_names,
                'facial_feature_obs_names' : facial_feature_obs_names,
                'sampling_obs_names' : sampling_obs_names,
                'valence_obs_names' : valence_obs_names,
                'facial_feature_action_names' : facial_feature_action_names,
                'sampling_action_names' : sampling_action_names,
                'valence_action_names' : valence_action_names
                
                }

    # timesteps at L1
    T=4

    # Generative process matrices -- wait, we don't actually seem to use these for the generative process, only the agent
    # -----------------------------

    A = utils.initialize_empty_A(num_obs, num_states)
    B = utils.initialize_empty_B(num_states=num_states, num_controls=num_controls) # B[0]:5*5*1 matrix B[1]
    C = utils.obj_array_zeros(num_obs) # no preferences
    D = utils.obj_array_zeros(num_states)

    # Facial Feature likelihoods, fill in. For SF1 there will be SF2*SF3 matrices, i.e. 2*2=4 matrices
    # A[SF1][OM1, SF1, SF2, SF3]
    # A[SF1][OM1, SF1: facial feature, SF2: sampling state , SF3: valence_state]
    # SF2_state = 0:sampling, SF3_state = 0:good
    #SF1   featS:  nul  eL  eR   mU    mD
    A[0][0,:,0,0]=[1,   0,  0,   0  ,  0 ] # null obs, agent will get null when it is in the 'stopping' state
    A[0][1,:,0,0]=[0,  .7,  .3, .0  , .0 ] # eye_left
    A[0][2,:,0,0]=[0 , .3,  .7, .0  , .0 ] # eye_right
    A[0][3,:,0,0]=[0 , .0 ,.0, .8  , .2 ] # mouth_up
    A[0][4,:,0,0]=[0 , .0 ,.0, .2  , .8] # mouth_down
    
    # A[SF1][OM1, SF1, SF2, SF3]
    # A[SF1][OM1, SF1: facial feature, SF2: sampling state , SF3: valence_state]
    # SF2_state = 0:sampling, SF3_state = 1:null
    #SF1   featS:  nul  eL  eR   mU    mD
    A[0][0,:,0,1]=[1,   0 ,  0,  0  ,  0 ] # null obs, agent will get null when it is in the 'stopping' state
    A[0][1,:,0,1]=[0,  .7,  .3, .0  , .0 ] # eye_left
    A[0][2,:,0,1]=[0 , .3,  .7, .0  , .0 ] # eye_right
    A[0][3,:,0,1]=[0 , .0 , .0, .8  , .2 ] # mouth_up
    A[0][4,:,0,1]=[0 , .0 , .0, .2  , .8 ] # mouth_down

    # A[SF1][OM1, SF1, SF2, SF3]
    # A[SF1][OM1, SF1: facial feature, SF2: sampling state , SF3: valence_state]
    # SF2_state = 1:stopped, SF3_state = 0:good
    #SF1   featS:  nul  eL  eR   mU    mD
    A[0][0,:,1,0]=[1,   0 , 0, 0  , 0 ] # null obs, agent will get null when it is in the 'stopping' state
    A[0][1,:,1,0]=[0,   1,  0, 0  , 0 ] # eye_left
    A[0][2,:,1,0]=[0 ,  0,  1, 0  , 0 ] # eye_right
    A[0][3,:,1,0]=[0 ,  0,  0, 1  , 0 ] # mouth_up
    A[0][4,:,1,0]=[0 ,  0,  0, 0  , 1 ] # mouth_down

    # A[SF1][OM1, SF1, SF2, SF3]
    # A[SF1][OM1, SF1: facial feature, SF2: sampling state , SF3: valence_state]
    # SF2_state = 1:stopped, SF3_state = 1:null    
    #SF1   featS:  nul  eL  eR   mU    mD
    A[0][0,:,1,1]=[1,   0 , 0 , 0  ,  0 ] # null obs, agent will get null when it is in the 'stopping' state
    A[0][1,:,1,1]=[0,   1,  0,  0  ,  0 ] # eye_left
    A[0][2,:,1,1]=[0 ,  0,  1,  0  ,  0 ] # eye_right
    A[0][3,:,1,1]=[0 ,  0 , 0,  1  ,  0 ] # mouth_up
    A[0][4,:,1,1]=[0 ,  0 , 0,  0  ,  1 ] # mouth_down

    # A[SF2][OM2, SF1, SF2, SF3]
    # Fill in Sampling Feedback Modality
    for feature in enumerate(facial_feature_state_names): # Cycle through SF1 states (facial features)
        #SF2          obs: nSam nSto
        A[1][:,feature[0],0,0]=[1, 0] # OM1: noticed sampling, SF1 = feature, SF2 = 0:sampling, SF3 = 0:valence good
        A[1][:,feature[0],0,1]=[1, 0] # OM1: noticed sampling, SF1 = feature, SF2 = 0:sampling, SF3 = 1:valence null
        A[1][:,feature[0],1,0]=[0, 1] # OM1: noticed stopping, SF1 = feature, SF2 = 0:stopping, SF3 = 0:valence good
        A[1][:,feature[0],1,1]=[0, 1] # OM1: noticed stopping, SF1 = feature, SF2 = 0:stopping, SF3 = 0:valence null

    # A[SF3][OM2, SF1, SF2, SF3] - For SF3 there are SF1*SF3 matrices, i.e. 5*2, the 5 are in the loop
    # A[SF3][OM2, SF1: facial feature, SF2: sampling state , SF3: valence_state]
    # SF1_state = looped, SF2_state = 0:sampling
    for feature in enumerate(facial_feature_state_names):
        #SF3           valS: good null  /OM2 Obs
        A[2][0,feature[0],0,:]=[0.7, 0.3] # noticing_v1, SF2=sampling
        A[2][1,feature[0],0,:]=[0.3, 0.7] # null, SF2=sampling
        A[2][0,feature[0],1,:]=[0.7, 0.3] # noticing_v1, SF2=stopping
        A[2][1,feature[0],1,:]=[0.3, 0.7] # null, SF2=stopping
    
    # A Learning
    pA = utils.dirichlet_like(A, scale = .3)

    # Make null observation contingencies 'un-learnable'
    # null is 'unavailable to learning'
    # SF1 OM1, all SF1, SF2,SF3
    pA[0][0,:,:,:] *= 10e5 # null

    # B[SF1][{facial_feature_state at tau+1}, {facial_feature_state at tau}, {CF1=facial_feature_action}]
    # agent assumes that hidden facial features don't change over time, i.e. they don't transition
    print(f'B[SF1].shape:{B[0].shape}')
    B[0][:,:,0] = np.eye(num_states[0]) # CF1_action=do_nothing
    B[0][:,:,0] = np.eye(num_states[0]) # CF1_action=do_nothing
    B[0][:,:,0] = np.eye(num_states[0]) # CF1_action=do_nothing
    B[0][:,:,0] = np.eye(num_states[0]) # CF1_action=do_nothing

    #B_choice / B_action
    #*********************
    # final value is the slice which is the chosen action

    # B[SF2][{SF2 at tau+1}, {SF2 at tau}, {{CF2=go_sample or stop_sampling}]
    # Sampling action moves from sampling state to sampling state
    # ------------
    # sampling -> [1, 0]
    B[1][0,0,0] = 1.0 # sampling to sampling B[1][sample to sample, stop to sample, go_sample]
    
    # Sampling action (doesn't) move from stopping state to sampling state
    B[1][0,1,0] = 0.0 # cannot move from stop to sampling
    
    # stopping -> [0, 0]
    B[1][1,0,0] = 0.0 # sampling to stopping no, b/c sampling action chosen
    B[1][1,1,0] = 1.0 # stop to stop only even if sampling action chosen
    
    # Stopping action
    # sampling -> [0, 1] no probability to move to sampling w/ stopping action ()
    B[1][0,0,1] = 0.0 # samp to samp No!
    B[1][0,1,1] = 0.0 # stop to samp No!

    # stopping -> [0, 1]
    B[1][1,0,1] = 1.0 # sampling to stop
    B[1][1,1,1] = 1.0 # stopping to stopping 

    # B[SF3][{SF3 at tau+1}, {SF3 at tau}, {{CF2=do_nothing}]
    # Valence action does nothing
    # ------------
    #              g n
    B[2][0,:,0] = [1,0] # expect move from 'good' to 'good' valence upon CF3='do_nothing' action
    B[2][1,:,0] = [0,1] # expect move from 'null' to 'null' valence upon CF3='do_nothing' action

    # C vector

    C[1][1] = reward_for_stopping # motivation to move toward breaking aka URGENCY
    
    if verbose:
        utils.plot_beliefs(C[1], title = "C[1] Prior prefs")
    
    # D matrix
    # SF1: Agent doesn't know what they are seeing, will need to consult env, regardless of valence state..
    D[0][0]=0
    D[0][1]=.25
    D[0][2]=.25
    D[0][3]=.25
    D[0][4]=.25

    # SF2: Agent believes they are in the sampling state
    D[1][0]=1
    D[1][1]=0

    # SF3: Agent believes that the L1 valence is 'good' to start with
    D[2][:]=[.2,.8]
    
    if verbose:
        utils.plot_beliefs(D[0], title = "D[0] Initial Facial Feature State Beliefs")
        utils.plot_beliefs(D[1], title = "D[1] Initial Sampling State Beliefs")
        utils.plot_beliefs(D[2], title = "D[2] Initial Valence State Beliefs")
    
    parameters = {'A': A,
            'B': B,
            'C': C,
            'D': D,
            'pA' : pA
            }
    return parameters, mappings, dimensions