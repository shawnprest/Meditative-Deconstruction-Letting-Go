from pymdp.agent import Agent
from pymdp.envs import Env
from pymdp import utils, maths, control, learning
import numpy as np
from copy import deepcopy

def init_L3_generative_model(pref_to_not_feel_bad=0, verbose=False):
  # L3 States, Observations, Actions, Matrices & Vectors:

  ###################### States ################################

  # SF1 valence context states
  # possible contexts
  context_state_names = ['normal', 'feel_bad' ]
  where_state_names = ['normal_location', 'defabricate_location' ]
  if verbose: print(f'context_state_names: {context_state_names}')
  
  ###################### Observations #############################

  # Obs modality 1: valence context obs
  # all possible things which can be observed (sent up from L2)
  context_obs_names = ['noticing_bad', 'noticing_neutral', 'noticing_good'] # these match L2 valence belief options
  if verbose: print(f'\ncontext_obs_names: {context_obs_names}')

  where_obs_names = ['noticing_normal_location', 'noticing_defabricate_location'] # these match L2 valence belief options
  if verbose: print(f'\ncontext_obs_names: {context_obs_names}')
  
  # Various L2 actions which can be taken
  where_action_names = ['do_nothing', 'choose_defabricate']  

  #################################################################
  # Package these lists

  mappings = { 'context_state_names' : context_state_names,
              'context_obs_names' : context_obs_names,
              'where_state_names' : where_state_names,
              'where_obs_names' : where_obs_names,
              'where_action_names' : where_action_names              
             }

  # Dimensions
  num_states = [len(context_state_names), len(where_state_names)] # (2)
  num_factors = len(num_states) # (1)
  num_obs = [len(context_obs_names), len(where_obs_names)] # (3)
  num_modalities = len(num_obs) # (1)
  num_controls = [1, len(where_action_names)]
  
  if verbose: print(f'num_states: {num_states}')
  if verbose: print(f'num_obs: {num_obs}')
  if verbose: print(f'num_controls: {num_controls}')
  
  dimensions = {  'num_states' : num_states,
                  'num_obs' : num_obs,
                  'num_controls' : num_controls }

  # Set up arrays for agent
  A = utils.initialize_empty_A(num_obs, num_states) # (OBS=3, S=2)
  B = utils.initialize_empty_B(num_states=num_states, num_controls=num_controls) # (S=2, S=2, C=2)
  C = utils.obj_array_zeros(num_obs) # (OBS=3)
  D = utils.obj_array_uniform(num_states) # (S=2)

  # ------------------------------------
  # A[SF1][OM1, SF1, SF2]
  # context_state_names = ['normal', 'defabricate' ]]
  
  #SF2: 'normal_loc':
  #SF1          nm   fb  # OM1    
  A[0][0,:,0]=[0.10, .60] # noticing_bad
  A[0][1,:,0]=[0.30, .30] # noticing_neutral
  A[0][2,:,0]=[0.60, .10] # noticing_good
  
  #SF2: 'defab_loc': #this implies we have just defabricated - nm / noticing_bad must be high enough that the agent doesnt defab every time - it is like a tolerance
  #SF1          nm   fb  # OM1    
  A[0][0,:,1]=[0.15, 0.40] # noticing_bad      in defab_loc with normal belief we expect to see neutral or good obs
  A[0][1,:,1]=[0.35, 0.35] # noticing_neutral  in defab_loc with feelbad belief we expect to see better beliefs than before
  A[0][2,:,1]=[0.50, 0.25] # noticing_good

  # in defab_loc with a 'normal' belief prior, we expect to see neutral or good obs
  # in defab_loc with 'feelbad' belief prior, we expect to see noticing_bad obs

  #--------------------------
  # A State Factor 2: location
  # A[SF2][OM2, SF1, SF2]
  # SF1='normal'
  #       SF2 nL dL
  A[1][0,0,:]=[1, 0] # noticing normal loc // if feeling normal (SF1) and the SF2 location is feel_bad, we should expect the normal location observation to move us to defabricate
  A[1][1,0,:]=[0, 1] # noticing defab loc // if feeling normal (SF1) and the SF2 location is 
  
  # if SF1='feel_bad'
  #       SF2 nL dL
  A[1][0,1,:]=[1, 0] # noticing normal loc // if feeling normal (SF1) and the SF2 location is feel_bad, we should expect the normal location observation to move us to defabricate
  A[1][1,1,:]=[0, 1] # noticing defab loc // if feeling normal (SF1) and the SF2 location is 

  A[0]=utils.norm_dist(A[0])
  A[1]=utils.norm_dist(A[1])
  if verbose: print(f'A[1].shape: {A[1].shape}')

  if verbose:
    # SF1
    utils.plot_likelihood(A[0][:,:,0], title = "A_L3[0][:,:,0]") # (SF2=normal_loc) 
    utils.plot_likelihood(A[0][:,:,1], title = "A_L3[0][:,:,1]") # (SF2=defab_loc)
    # SF2
    utils.plot_likelihood(A[1][:,0,:], title = "A_L3[1][:,0,:]") # (SF1=normal) 
    utils.plot_likelihood(A[1][:,1,:], title = "A_L3[1][:,1,:]") # (SF1=feel_bad) 
  
  
  # Transition (B) Matrix *****************************************************************
  # for example: B[0].shape is (num_states[0], num_states[0], num_controls[0])
  # thus each slice is a different action

  # agent assumes that the hidden face type doesnt change over time
  # B[SF1][{SF1 at tau+1}, {SF1 at tau}, {{CF2=do_nothing,choose_defabricate}]

  # B-matrix for 'do_nothing'
  #              nm df
  B[0][:,:,0] = np.eye(len(context_state_names))

  # B SF2 matrix for action 'do_nothing'
  #              nm df
  B[1][0,:,0] = [1, 1] # to 'normal'
  B[1][1,:,0] = [0, 0] # to 'defabricate'

  # B SF2 matrix for action 'choose_defabricate'
  #              nm df
  B[1][0,:,1] = [0, 0] # to 'normal'
  B[1][1,:,1] = [1, 1] # to 'defabricate'

  if verbose: #B[SF1][next, current, action]
    utils.plot_likelihood(B[1][:,:,0], title = "B_L3[SF2][:,:,0] SF1 do_nothing")
    utils.plot_likelihood(B[1][:,:,1], title = "B_L3[SF2][:,:,1] SF1 choose_defabricate")

  # C Matrix Preferences
  
  #What preferences for certain observations (OM1 - context)
  # context_obs_names = ['noticing_bad', 'noticing_neutral', 'noticing_good']
  #        bad  ntrl good
  C[0][:]=[pref_to_not_feel_bad, 0.0, 0] # to motivate defab, bad is -2
  
  if verbose:
    utils.plot_beliefs(C[0][:], title = "C_L3[SF1][:]") # OM1 - Context

  #SF1     norm  fb
  D[0][:]=[0.5, 0.5]

  #SF2
  #       nL  dL
  D[1][:]=[.8, .2]

  if verbose:
    utils.plot_beliefs(D[0][:], title = "D_L3[SF1][:]") # context
    utils.plot_beliefs(D[1][:], title = "D_L3[SF2][:]") # feedback

  parameters = {'A': A,
                'B': B,
                'C': C,
                'D': D
                }
  
  return parameters, mappings, dimensions

#end