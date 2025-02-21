# Utility functions for multi-layer agent

from pymdp import utils, maths, control, learning
import numpy as np
import maths_custom as maths2


# Send L1 beliefs to L2 observations
def get_obs_from_below(beliefs_from_below=None, where_obs_from_above=None, valence_obs_from_below=None, dimensions_from_above=None, mappings_from_above=None, verbose=False):

    # we send up the belief from below with the location from above we already knew    
    if verbose: print(f'GET_OBS_FROM_BELOW: beliefs_from_below: {beliefs_from_below}')
    if verbose: print(f'GET_OBS_FROM_BELOW: where_obs_from_above: {where_obs_from_above}')
    if verbose: print(f"GET_OBS_FROM_BELOW: mappings_from_above['where_obs_names']: {mappings_from_above['where_obs_names']}")
    # make array the size of all above observations
    obs_to_above = utils.obj_array_zeros(dimensions_from_above['num_obs'])
    # dump the beliefs from below in the appropriate part of it
    obs_to_above[0][1:5] = beliefs_from_below[0][1:] # for SF1, remove null.
    obs_to_above[2][0] = beliefs_from_below[2][0] # for SF3 (valence:good) convert to noticing_v1 prob
    obs_to_above[2][1] = beliefs_from_below[2][1]/2.5 # for SF3 (valence:neutral) convert to noticing_v2 prob
    obs_to_above[2]=utils.norm_dist(obs_to_above[2])
    if verbose: print(f'GET_OBS_FROM_BELOW what: obs_from_above[0][3:] = beliefs_from_below[0][1:]: {obs_to_above[0][1:5]}\tvalence:{obs_to_above[2][:]}')
    # get index position for the particular observation from above
    where_obs_to_above_idx = mappings_from_above['where_obs_names'].index(where_obs_from_above) # containts the tertiant observation
    if verbose: print(f'GET_OBS_FROM_BELOW: where_obs_from_above_idx: {where_obs_to_above_idx}')

    # insert a one for the correct where_obs location
    # i.e. in 'null', 'face_happy', 'face_sad', 'eye_left', 'eye_right' etc...
    obs_to_above[1] = utils.onehot(where_obs_to_above_idx, dimensions_from_above['num_obs'][1])
    if verbose: print(f'GET_OBS_FROM_BELOW: obs_from_above variable:\twhat {obs_to_above[0]}\twhere: {obs_to_above[1]}')
    return obs_to_above

# Send L2 beliefs up as L3 observations
def get_obs_from_L2(beliefs_from_below=None, dimensions_from_above=None, mappings_from_above=None, verbose=False):
    # make array the size of all above observations
    obs_to_above = utils.obj_array_zeros(dimensions_from_above['num_obs'])
    # dump the beliefs from below in the appropriate part of it, but increase their precision first to make L3 track L2 valence more dynamically
    precise_where_beliefs_from_below=maths2.softmax_precision(beliefs_from_below[2], 20.0)
    obs_to_above[0] = precise_where_beliefs_from_below # {'bad','neutral','good'}->{'noticing_bad','noticing_neutral','noticing_good'}
    #obs_to_above[1]=np.argmax(beliefs_from_below[1]) # This returns 0 if bad, 1 if neut, 2 if good
    obs_to_above[1][:]=[1.0, 0.0] # set L3 back to 'noticing_normal_location' whether it was previously on 'noticing_normal_location' or 'noticing_defabricate_location' prior to L2 inference    
    return obs_to_above

# Send L2 expected obs to L1 belief (empirical prior D)
def get_prior_from_above_classic(beliefs_high=None, A_above=None, dimensions_low=None, verbose=False):
    # map `q_states` through the likelihood mapping to make a prior over the hidden states of the lower-level agent
    expected_what_obs = maths.spm_dot(A_above[0], beliefs_high) # SF1: what_state
    expected_valence_obs = maths.spm_dot(A_above[2], beliefs_high) # SF3: valence_state->OM3 valence obs {bad neutral good}->{noticing_v1, noticing_v2}
    if verbose: print(f'L2->L1 Link: expected valence obs:{expected_valence_obs}\texpected_valence_obs:{expected_valence_obs}')
    empirical_prior = utils.obj_array_zeros(dimensions_low['num_states']) # make an empirical prior over hidden states at the low-level
    #if verbose: print(f'L2->L1 Link: empirical prior:{empirical_prior}')
    empirical_prior[0] = expected_what_obs[:5]
    # expected_valence_obs[0] is the probability of noticing_v1, expected_valence_obs[1] is the prob of noticing_v2
    #                                     SF3_L1: g   n

    empirical_prior[2][0] = expected_valence_obs[0] # noticing_v1->valenceL1=good
    empirical_prior[2][1] = expected_valence_obs[1] # noticing_v2->valenceL1=neutral
    
    #empirical_prior[0][0] = 0.0
    #if verbose: print(f'L2->L1 Link: empirical prior with zero:{empirical_prior}')
    empirical_prior[1] = utils.onehot(0, dimensions_low['num_states'][1]) # For sampling prior
    if verbose: print(f'\tL2->L1 Link: empirical prior:\n\twhat: {np.round(empirical_prior[0], 2)}\n\twhere: {np.round(empirical_prior[1], 2)}\n\tvalence: {np.round(empirical_prior[2], 2)}\n')
    return empirical_prior

# Send L3 expected context obs to L2 empirical prior
def get_valence_context_prior_from_L3(beliefs_high=None, A_above=None, dimensions_low=None, verbose=False): 
    expected_context_obs = maths.spm_dot(A_above[0], beliefs_high) # calc expected higher-level observation from likelihood and current higher-level beliefs
    if verbose: print(f'L3->L2 Link: L3 expected valence context obs:{expected_context_obs}') # [bad=1, neutr=0, good=0] - if bad is predicted then one hot
    #empirical_prior = utils.obj_array_zeros(dimensions_low['num_states']) # create lower level SF2 empirical prior structure for lower level hidden states
    where_idx=np.argmax(expected_context_obs) # This returns 0 if bad, 1 if neut, 2 if good
    if where_idx==0: # bad belief
        empirical_where_prior = utils.onehot(where_idx+6, dimensions_low['num_states'][1]) # Send down expectation for OM2='defabrication_understanding'
        empirical_where_prior[:]=[0,.333,.333,.333,0,0,0]
    elif where_idx==1: # neut belief
        empirical_where_prior = utils.onehot(where_idx-1, dimensions_low['num_states'][1]) # Send down expectation for OM2='seeing_eyelids'
        empirical_where_prior[:]=[0,0,0,0,.5,.5,0] # change to fH fS prior
    elif where_idx==2: # good belief
        empirical_where_prior = utils.onehot(where_idx-2, dimensions_low['num_states'][1]) # Send down expectation for OM2='seeing_eyelids'
        empirical_where_prior[:]=[0,0,0,0,.5,.5,0] # change to fH fS prior
    if verbose: print(f'L3->L2 Link: L2 empirical prior:{np.round(empirical_where_prior, 2)}')
    return empirical_where_prior