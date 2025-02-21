from pymdp.agent import Agent
from pymdp.envs import Env
from pymdp import utils, maths, control, learning
import numpy as np
from copy import deepcopy

def init_V2_generative_model(T=3, reward_for_correct=2, reward_for_incorrect=-0.4, reward_for_defabricate=0.0, reward_for_bad=0.0, reward_for_good=0.0, verbose=False):
  # V2 Environmental Aspects

  ###################### states ################################
  # SF1 what states
  # possible what_states: SF1
  what_state_dict = { 'tert1_eL_time1': 0, 'tert1_eL_time2' : 1, 'tert1_eL_time3' : 2,
   'tert2_mU_time1' : 3, 'tert2_mD_time1': 4, 'tert2_mU_time2' : 5, 'tert2_mD_time2' : 6, 'tert2_mU_time3' : 7, 'tert2_mD_time3' : 8,
   'tert3_eR_time1' : 9, 'tert3_eR_time2' : 10, 'tert3_eR_time3' : 11}
  
  # possible scenes
  what_state_names = ['1.eL_2.mU_3.eR', '1.eL_2.mD_3.eR' ]
  
  if verbose: print(f'what_state_names: {what_state_names}')

  # SF2 where states
  # these are what the agent can be comprehending at a given moment
  # the tertiant_names imply going down to sample that tertiant via V1
  where_state_names = ['eyelids', 'tertiant_1','tertiant_2','tertiant_3', 'face_happy_understanding', 'face_sad_understanding', 'defabricate_understanding'] # the (proprioceptive) eye position 
  if verbose: print(f'where_state_names: {where_state_names}')

  # SF3 valence states
  valence_state_names = ['bad', 'neutral', 'good'] 

  ###################### observations #############################
  # Obs modality 1: what obs
  # all possible things which can be observed at V2
  what_obs_names = ['null', 'seeing_eye_left', 'seeing_eye_right', 'seeing_mouth_up', 'seeing_mouth_down', 'correct', 'incorrect' ,'seeing_defabricate']
  if verbose: print(f'\nwhat_obs_names: {what_obs_names}')

  # Obs modality 2: where obs
  # Eye position - eyes can be on lids, tertiants or broader field (a particular grouping)
  where_obs_names = ['seeing_eyelids', 'seeing_tertiant_1', 'seeing_tertiant_2', 'seeing_tertiant_3', 'seeing_face_happy_understanding', 'seeing_face_sad_understanding', 'seeing_defabricate_understanding']
  if verbose: print(f'where_obs_names: {where_obs_names}')

  # Obs modality 3: valence obs
  # affective valence observation
  valence_obs_names = ['noticing_v1','noticing_v2'] # valence observations associated with inference at levels 1 ('noticing_v1') or 2 ('noticing_v2')
  
  ###################### actions #################################
  # Various L2 actions which can be taken
  where_action_names = ['choose_eyelids', 'inspect_tertiant_1', 'inspect_tertiant_2', 'inspect_tertiant_3', 'choose_face_happy', 'choose_face_sad', 'choose_defabricate']
  valence_action_names=['do_nothing']
  if verbose: print(f'action_names: SF1:{where_action_names}\tSF3:{valence_action_names}')

  #################################################################
  # package these lists

  mappings = { 'what_state_names' : what_state_names,
              'where_state_names' : where_state_names,
              'valence_state_names' : valence_state_names,
              'what_obs_names' : what_obs_names,
              'where_obs_names' : where_obs_names,
              'valence_obs_names' : valence_obs_names,
              'action_names' : where_action_names,
              'valence_action_names' : valence_action_names
              
             }

  # Only two possible uniquely L2 states (happy + sad) plus the various locations the eyes can be in
  # Here is where I would add all of the other possible permutations which do not result in seeing a face
  # i.e. t1=eye_right, t2=eye_left, t3=mouth_down
  num_states = [len(what_state_names), len(where_state_names), len(valence_state_names)] # (12, 6, 3)
  num_factors = len(num_states) # (3)
  num_obs = [len(what_obs_names), len(where_obs_names), len(valence_obs_names)] # (7, 6, 2)
  
  num_controls = [1, len(where_action_names), len(valence_action_names)] # CF (1, 7, 2) # there are three control factors with 1 state (what - hardcoded), 5 states (where), and 1 state (valence - hardcoded)
  num_modalities = len(num_obs)
  if verbose: print(f'num_states: {num_states}')
  if verbose: print(f'num_obs: {num_obs}')
  if verbose: print(f'num_controls: {num_controls}')
  
  dimensions = {  'num_states' : num_states,
                  'num_obs' : num_obs,
                  'num_controls' : num_controls }

  # Set up arrays for agent
  A = utils.initialize_empty_A(num_obs, num_states)
  B = utils.initialize_empty_B(num_states=num_states, num_controls=num_controls)  
  C = utils.obj_array_zeros(num_obs) # (5, 6)
  D = utils.obj_array_uniform(num_states) # (12, 6, 3)

  # Fill in What Obs/States Modality and set to A[0]
  # When agent is in where_state: eyelids, observations cannot be anything other than null regardless of what it thinks its looking at
  
  # A[SF1][OM1, SF1, SF2, SF3]
  # what_state_names = ['1.eL_2.mU_3.eR', '1.eL_2.mD_3.eR' ]]
  for valence_state in enumerate(valence_state_names): # loop for A matrix through all of the valence states, valence doesn't alter perception of what and where
    # SF2 (location) = EYELIDS / NULL, SF3 (valence)
    #                  SF1       fH fS  # OM1
    A[0][0,:,0,valence_state[0]]=[1, 1] # seeing eyelids / null / start
    A[0][1,:,0,valence_state[0]]=[0, 0] # seeing eye_left
    A[0][2,:,0,valence_state[0]]=[0, 0] # seeing eye_right
    A[0][3,:,0,valence_state[0]]=[0, 0] # seeing mouth_up
    A[0][4,:,0,valence_state[0]]=[0, 0] # seeing mouth_down
    A[0][5,:,0,valence_state[0]]=[0, 0] # correct
    A[0][6,:,0,valence_state[0]]=[0, 0] # incorrect
    A[0][7,:,0,valence_state[0]]=[0, 0] # seeing_defabricate      

    # SF2 (location) = TERTIANT 1 - we can think about these values as 'noise' in visual perception
    #                            fH  fS  
    A[0][0,:,1,valence_state[0]]=[0, 0] # seeing eyelids / null / start
    A[0][1,:,1,valence_state[0]]=[.55, .55] # seeing eye_left .7 old
    A[0][2,:,1,valence_state[0]]=[.15, .15] # seeing eye_right .3 old
    A[0][3,:,1,valence_state[0]]=[.15, .15] # seeing mouth_up
    A[0][4,:,1,valence_state[0]]=[.15, .15] # seeing mouth_down
    A[0][5,:,1,valence_state[0]]=[0, 0] # correct
    A[0][6,:,1,valence_state[0]]=[0, 0] # incorrect
    A[0][7,:,1,valence_state[0]]=[0, 0] # seeing_defabricate

    # SF2 (location) = TERTIANT 2
    #                            fH  fS
    A[0][0,:,2,valence_state[0]]=[0, 0] # seeing eyelids / null / start
    A[0][1,:,2,valence_state[0]]=[.15, .15] # seeing eye_left
    A[0][2,:,2,valence_state[0]]=[.15, .15] # seeing eye_right
    A[0][3,:,2,valence_state[0]]=[.55, .15] # seeing mouth_up .7 .3 old
    A[0][4,:,2,valence_state[0]]=[.15, .55] # seeing mouth_down .3 .7 old
    A[0][5,:,2,valence_state[0]]=[0, 0] # correct
    A[0][6,:,2,valence_state[0]]=[0, 0] # incorrect
    A[0][7,:,2,valence_state[0]]=[0, 0] # seeing_defabricate

    # SF2 (location) = TERTIANT 3
    #                            fH  fS
    A[0][0,:,3,valence_state[0]]=[0, 0] # seeing eyelids / null / start
    A[0][1,:,3,valence_state[0]]=[.15, .15] # seeing eye_left # 0.3 old
    A[0][2,:,3,valence_state[0]]=[.55, .55] # seeing eye_right # 0.7 old
    A[0][3,:,3,valence_state[0]]=[.15, .15] # seeing mouth_up
    A[0][4,:,3,valence_state[0]]=[.15, .15] # seeing mouth_down
    A[0][5,:,3,valence_state[0]]=[0, 0] # correct
    A[0][6,:,3,valence_state[0]]=[0, 0] # incorrect
    A[0][7,:,3,valence_state[0]]=[0, 0] # seeing_defabricate

    # SF2 (location) = FACE_HAPPY_UNDERSTANDING
    #                            fH  fS
    A[0][0,:,4,valence_state[0]]=[0, 0] # seeing eyelids / null / start
    A[0][1,:,4,valence_state[0]]=[0, 0] # seeing eye_left
    A[0][2,:,4,valence_state[0]]=[0, 0] # seeing eye_right
    A[0][3,:,4,valence_state[0]]=[0, 0] # seeing mouth_up
    A[0][4,:,4,valence_state[0]]=[0, 0] # seeing mouth_down
    A[0][5,:,4,valence_state[0]]=[1, 0] # correct
    A[0][6,:,4,valence_state[0]]=[0, 1] # incorrect
    A[0][7,:,4,valence_state[0]]=[0, 0] # seeing_defabricate

    # SF2 (location) = FACE_SAD_UNDERSTANDING
    #                            fH  fS
    A[0][0,:,5,valence_state[0]]=[0, 0] # seeing eyelids / null / start
    A[0][1,:,5,valence_state[0]]=[0, 0] # seeing eye_left
    A[0][2,:,5,valence_state[0]]=[0, 0] # seeing eye_right
    A[0][3,:,5,valence_state[0]]=[0, 0] # seeing mouth_up
    A[0][4,:,5,valence_state[0]]=[0, 0] # seeing mouth_down
    A[0][5,:,5,valence_state[0]]=[0, 1] # correct
    A[0][6,:,5,valence_state[0]]=[1, 0] # incorrect
    A[0][7,:,5,valence_state[0]]=[0, 0] # seeing_defabricate
  
    # SF2 (location) = defabricate_understanding
    # If the location is SF2=defabricate_understanding then we expect to see 'seeing_defabricate'
    # If the valence_state is SF3=bad then we expect (prefer) to see 'seeing_defabricate'
    #                             fH  fS
    A[0][0,:,6,valence_state[0]]=[0.0, 0.0] # seeing eyelids / null / start
    A[0][1,:,6,valence_state[0]]=[0.0, 0.0] # seeing eye_left
    A[0][2,:,6,valence_state[0]]=[0.0, 0.0] # seeing eye_right
    A[0][3,:,6,valence_state[0]]=[0.0, 0.0] # seeing mouth_up
    A[0][4,:,6,valence_state[0]]=[0.0, 0.0] # seeing mouth_down
    A[0][5,:,6,valence_state[0]]=[0.0, 0.0] # correct
    A[0][6,:,6,valence_state[0]]=[0.0, 0.0] # incorrect
    A[0][7,:,6,valence_state[0]]=[1.0, 1.0] # seeing_defabricate

  # For SF3={0,1,2} the likelihoods are all the same
  if verbose:
    utils.plot_likelihood(A[0][:,:,0,0], title = "A_L2[0][:,:,0,0]") # (SF2=eyelids, SF3=bad) 
    utils.plot_likelihood(A[0][:,:,1,0], title = "A_L2[0][:,:,1,0]") # (SF2=tertiant_1, SF3=bad) 
    utils.plot_likelihood(A[0][:,:,2,0], title = "A_L2[0][:,:,2,0]") # (SF2=tertiant_2, SF3=bad)
    utils.plot_likelihood(A[0][:,:,3,0], title = "A_L2[0][:,:,3,0]") # (SF2=tertiant_3, SF3=bad)
    utils.plot_likelihood(A[0][:,:,4,0], title = "A_L2[0][:,:,4,0]") # (SF2=face_happy_understanding, SF3=bad)
    utils.plot_likelihood(A[0][:,:,5,0], title = "A_L2[0][:,:,5,0]") # (SF2=face_sad_understanding, SF3=bad)
    utils.plot_likelihood(A[0][:,:,6,0], title = "A_L2[0][:,:,6,0]") # (SF2=defabricate_understanding, SF3=bad)
  
  # Obs Modality 2: 
  # these are all identity matrices, thus there is no uncertainty about locations
  # --------------------------------------------------------------------------
  # where_obs (OM2), where_state (SF2) WHEN IN 'what_state' (SF1) = null
  # assuming we have no understood relationship between different tertiants and different observations initially..
  
  # A[SF2][OM2, SF1, SF2, SF3]
  for i in range(len(what_state_names)):
    for valence_state in enumerate(valence_state_names): # loop for all valence states, bad valence means defabrication observations are expected
    #                     SF2 (where) lid  t1   t2   t3   fHu  fSu  dU
      A[1][0,i,:,valence_state[0]]=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # eyelid obs (OM1) (OM1=0, SF1=0, SF2 = [1 0 0 0 0])  when you see eyelids, what = nothing & where = eyelids
      A[1][1,i,:,valence_state[0]]=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] # tertiant_1 obs - when you're seeing tertiant 1, what = nothing & where = tertiant_1
      A[1][2,i,:,valence_state[0]]=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] # tertiant_2 obs
      A[1][3,i,:,valence_state[0]]=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] # tertiant_3 obs
      A[1][4,i,:,valence_state[0]]=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] # seeing_face_happy_understanding
      A[1][5,i,:,valence_state[0]]=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] # seeing_face_sad_understanding
      A[1][6,i,:,valence_state[0]]=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # seeing_defabricate_understanding       
  
  # A[SF3][OM3, SF1, SF2, SF3]
  for what_state in enumerate(what_state_names):
    for where_state in enumerate(where_state_names):
      #                                        bad  ntrl good
      A[2][0,what_state[0],where_state[0],:]=[0.45, 0.50, 0.60] # OM3=noticing_v1, SF3={bad,neutral,good,defab_please} # old: 0.45, 0.50, 0.66
      A[2][1,what_state[0],where_state[0],:]=[0.55, 0.50, 0.40] # OM3=noticing_v2, SF3={bad,neutral,good,defab_please} #    : 0.55, 0.50, 0.34

  if verbose:
    utils.plot_likelihood(A[2][:,0,0,:], title = "A_L2[2][:,0,0,:]") # (SF1=eyelids, SF2=eyelids) 

  A[0]=utils.norm_dist(A[0])
  A[1]=utils.norm_dist(A[1])
  A[2]=utils.norm_dist(A[2])
  
  # A Matrix Learning - currently there is only Matrix Learning for A matrix likelihoods!
  pA = utils.dirichlet_like(A, scale = 0.4)
  for valence_state in enumerate(valence_state_names): # loop for all valence states
    pA[0][0,:,:,valence_state[0]] *= 10e5 # SF1 - OM1 is observing null
    pA[1][0,:,:,valence_state[0]] *= 10e5 # SF2 - OM2 is observing eyelids
    #pA[2][0,:,:,valence_state[0]] *= 10e5 # SF2 - OM2 is observing eyelids
  # pA[SF3][OM3:{noticing_v1, noticing_v2},what_state,where_state,valence_state]
  
  pA[2][:,:,:,:] *= 10e5

  # Transition (B) Matrix *****************************************************************
  # for example: B[0].shape is (num_states[0], num_states[0], num_controls[0])
  # thus each slice is a different action
  
  # agent assumes that the hidden face type doesnt change over time
  # B[SF3][{SF3 at tau+1}, {SF3 at tau}, {{CF2=do_nothing}]
  control_fac_idx = [1]
  for f, ns in enumerate(num_states):
      if f in control_fac_idx:
          B[f] = utils.construct_controllable_B( [ns], [num_controls[f]] )[0]
      else:
          B[f][:,:,0] = np.eye(ns)


  # B[SF2] - for choose_defabricate action
  # Make do nothing uncontrolable
  # i.e. np(eye(num_states))
  #              b  n  g
  B[2][0,:,0] = [1, 0, 0] # to bad
  B[2][1,:,0] = [0, 1, 0] # to neutral
  B[2][2,:,0] = [0, 0, 1] # to good

  if verbose: #B[SF1][next, current, action]
    #utils.plot_likelihood(B[0][:,:,0], title = "B_L2[0][:,:,0] do_nothing") # Tertiant 1 (SF2=tertiant_1, SF3=bad) 
    #utils.plot_likelihood(B[1][:,:,6], title = "B_L2[0][:,:,6] choose_defabricate") # Tertiant 1 (SF2=tertiant_1, SF3=neutral)
    #utils.plot_likelihood(B[0][:,:,0], title = "B_L2[1][:,:,0] SF2 do_nothing") # Tertiant 1 (SF2=tertiant_1, SF3=neutral)
    utils.plot_likelihood(B[2][:,:,0], title = "B_L2[2][:,:,0] SF3 do_nothing") # 
    #utils.plot_likelihood(B[2][:,:,1], title = "B_L2[2][:,:,1] SF3 choose_defabricate") # 
    #utils.plot_likelihood(B[0][:,:,2], title = "B_L2[0][:,:,2]") # Tertiant 1 (SF2=tertiant_1, SF3=good)

  if verbose:
    print(f'V2 INIT: B[0].shape: {B[0].shape}')
    print(f'V2 INIT: B[1].shape: {B[1].shape}')

  # C Matrix Preferences
  
  #What preferences for certain observations (OM1 - what)
  # ['null', 'seeing_eye_left', 'seeing_eye_right', 'seeing_mouth_up', 'seeing_mouth_down', 'correct', 'incorrect', 'seeing_defabricate'] # maybe add correct and incorrect?
  C[0][:]=[-0.1,0.0,0.0,0.0,0.0, reward_for_correct, reward_for_incorrect, reward_for_defabricate]
  
  #Where preferences for certain observations (OM2- why) - intrinsic drive to prefer under
  # 'seeing_eyelids', 'seeing_tertiant_1', 'seeing_tertiant_2', 'seeing_tertiant_3', 'seeing_face_happy_understanding', 'seeing_face_sad_understanding', 'seeing_defabricate_loc']
  C[1][:]=[0.0,0.0,0.0,0.0,0.0,0.0, 0]

  #Valence preferences for certain observation (OM3 - valence)
  # 'noticing_v1', 'noticing_v2'
  C[2][:]=[reward_for_good, reward_for_bad] # negative reward for feeling bad

  if verbose:
    utils.plot_beliefs(C[0][:], title = "C_L2[SF1][:]") # OM1 - What
    utils.plot_beliefs(C[1][:], title = "C_L2[SF2][:]") # OM2 - Where
    utils.plot_beliefs(C[2][:], title = "C_L2[SF3][:]") # OM3 - Valence

  if verbose:
    utils.plot_beliefs(D[0][:], title = "D_L2[SF1][:]") # SF1 - What
    utils.plot_beliefs(D[1][:], title = "D_L2[SF2][:]") # SF2 - Where
    utils.plot_beliefs(D[2][:], title = "D_L2[SF3][:]") # SF3 - Valence

  print(f'D[SF3]: {D[2]}')
  parameters = {'A': A,
                'B': B,
                'C': C,
                'D': D,
                'pA' : pA
                }
  
  return parameters, mappings, dimensions

#end