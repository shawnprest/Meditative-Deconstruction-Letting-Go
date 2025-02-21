from pymdp.envs import Env
from pymdp import utils, maths, control, learning
import numpy as np
import maths_custom as maths2
import random as random # used for lower level choice after defabrication
from copy import deepcopy

# L3 Env
class ValenceContext(Env):
  def __init__(self, precision=1.0, starting_context = 'normal', starting_location = None, mappings=None, verbose=False): # tertiant_contents is now an array with gen_process timestep changes
    self.precision=precision
    self.verbose=verbose
    self.context = starting_context # This actually isn't used, can probably remove later, the Env only tracks the state of (having chosen to) initiate defabricate' which is locational
    if starting_location == None:
      self.starting_location = 'normal_location'
    else:
      self.starting_location=starting_location
    self.location = self.starting_location
    self.current_where_obs=utils.onehot(0, 2) # idx pos 1 and two values #'noticing_normal_location'
    
    if mappings is None:
      print(f'Error, no mappings provided for the object labels!')
      # here we could potentially provide numbered mappings instead?
 
  def step(self, action):
    if action == 'do_nothing':
      if self.verbose: print(f'L3 Env action do_nothing')
      where_obs='noticing_normal_location'
      obs_idx_where = utils.onehot(0, 2) # idx pos 0 and two values
      # add observations here
    elif action == 'choose_defabricate':
      where_obs='noticing_defabricate_location'
      obs_idx_where = utils.onehot(1, 2) # idx pos 1 and two values
      if self.verbose: print(f'L3 Env action choose_defabricate')
      #A, beliefs = self.defabricate_A(A, dimensions, beliefs)
    else:
      print(f'Error, invalid action!')
    self.current_where_obs=obs_idx_where
    return self.current_where_obs # Return the L3 indication that the agent has initiated defabrication alongside reduced precision L2 A and qs distributions
  
  def defabricate_A(self, A, dimensions, beliefs):
    print('defabricate_A')
    Mode = 'beliefs_and_A_softmax' # ['empty', 'softmax', 'double_softmax', 'triple_softmax', 'beliefs_softmax', 'beliefs_and_A_softmax']
    if Mode == 'empty':
      A_defab = utils.initialize_empty_A(dimensions['num_obs'], dimensions['num_states'])
      return A_defab, beliefs
    elif Mode == 'softmax':
      A_defab_0=maths.softmax(A[0])
      A_defab_1=maths.softmax(A[1])
      return [A_defab_0, A_defab_1], beliefs
    elif Mode == 'double_softmax':
      A_defab_0=maths.softmax(A[0])
      A_defab_1=maths.softmax(A[1])
      A_defab_0=maths.softmax(A_defab_0.copy())
      A_defab_1=maths.softmax(A_defab_1.copy())
      return [A_defab_0, A_defab_1], beliefs
    elif Mode == 'triple_softmax':
      A_defab_0=maths.softmax(A[0])
      A_defab_1=maths.softmax(A[1])
      A_defab_0=maths.softmax(A_defab_0.copy())
      A_defab_1=maths.softmax(A_defab_1.copy())
      A_defab_0=maths.softmax(A_defab_0.copy())
      A_defab_1=maths.softmax(A_defab_1.copy())
      return [A_defab_0, A_defab_1], beliefs
    elif Mode == 'beliefs_softmax':
      beliefs_0=np.array(beliefs[0])
      beliefs_1=np.array(beliefs[1])
      beliefs[0] = maths2.softmax_precision(beliefs_0, 1)
      beliefs[1] = maths2.softmax_precision(beliefs_1, 1)
      A_defab_0=A[0]
      A_defab_1=A[1]
      return [A_defab_0, A_defab_1], beliefs
    elif Mode == 'beliefs_and_A_softmax': 
      beliefs_0=np.array(beliefs[0])
      beliefs_1=np.array(beliefs[1])
      beliefs[0] = maths2.softmax_precision(beliefs_0, 0.7)
      beliefs[1] = maths2.softmax_precision(beliefs_1, 0.7)
      A_defab_0=maths2.softmax_precision(A[0], 0.9)
      A_defab_1=maths2.softmax_precision(A[1], 0.9)
      return [A_defab_0, A_defab_1], beliefs
    else:
      print('Error in L3 env: defabricate_A function!')

# L2 Env
class Faces(Env):
  def __init__(self, precision=1.0, starting_location = None, tertiant_contents = None, mappings=None, verbose=False): # tertiant_contents is now an array with gen_process timestep changes
    self.precision=precision
    self.verbose=verbose
    self.starting_location=starting_location
    self.valence_state = 'v2' # bad is the default L2 valence
    self.valence_obs = 'noticing_v2'
    self.mappings=mappings
    self.L2_timestep=0

    if starting_location is None: #if no starting location then set to 'eyelids'
      self.starting_location = 'eyelids'
    else:
      self.starting_location = starting_location
        
    if mappings is None:
      print(f'Error, no mappings provided for the object labels!')
      # here we could potentially provide numbered mappings instead?
    
    self.tertiants=tertiant_contents # Load tertiant data for all timesteps, tertiant_contents[timestep][tertiant_number]
    self.current_location = self.starting_location
    self.context='normal' # 'normal' or 'defabricate'
    if verbose: print(f'Initial Env location: {self.current_location}')
    self.current_obs = 'null'
    self.current_valence = self.valence_state
  
  def step(self, action, valence_action, A, dimensions, beliefs):
    # update the features present in the agent's experience!
    self.L2_timestep = self.L2_timestep + 1
    print(f'env L2_time_updated: {self.L2_timestep}')
    # possible actions: ['inspect_tertiant_1', 'inspect_tertiant_2', 'inspect_tertiant_3', 'choose_face_happy', 'choose_face_sad', 'defabricate']

    location = self.current_location
    obs = self.current_obs
    if self.valence_state == 'v2':
      self.valence_obs == 'noticing_v2'
  
    if self.context=='defabricate': action='choose_defabricate'
    if action == 'choose_defabricate': # defabricate decision overrides CF2 actions
      new_location=random.choice(['tertiant_1','tertiant_2','tertiant_3'])
      action='inspect_' + new_location
      A, beliefs = self.defabricate_A(A, dimensions, beliefs)
    elif action == 'inspect_tertiant_1':
      new_location='tertiant_1'
    elif action == 'inspect_tertiant_2':
      new_location = 'tertiant_2'
    elif action == 'inspect_tertiant_3':
      new_location = 'tertiant_3'
    elif action == 'choose_face_happy':
        new_location = 'face_happy_understanding' # this is kind of like assimilating after looking around
    elif action == 'choose_face_sad':
      new_location = 'face_sad_understanding'
    else: # this is the do nothing case... should not ever occur
      print('Error in V2.step!')
      new_location = location
    self.current_location = new_location # update generative model location (if defab then it has chosen a random tertiant)
    if self.verbose: print(f'\nL2 step: returning new location: [{self.current_location}]')
    
    # Get observations based on chosen action to send to agent for what it sees and where (generative model)
    obs = self.get_observation(action)
    self.current_obs = obs
    if self.verbose: print(f'\nL2 step: returning what_obs: [{obs[0]}]\twhere_obs: [{obs[1]}]')
    return [self.current_obs[0], self.current_obs[1], self.current_obs[2], A, beliefs]

  def reset(self):
    self.current_location = self.starting_location
    if self.verbose: print(f'L2: Re-init where state (location) to [{self.current_location}]')
    what_obs = 'null'
    where_obs = self.mappings['where_obs_names'][0]
    if self.verbose: print(f'L2 reset: what_obs:{what_obs}\twhere_obs:{where_obs}')
    valence_obs='noticing_v2'
    return what_obs, where_obs, valence_obs

  def get_observation(self, action):
    
    #   what_obs_names: ['null', 'seeing_eye_left', 'seeing_eye_right', 'seeing_mouth_up', 'seeing_mouth_down', 'seeing_defabricate']
    #   where_obs_names: ['seeing_eyelids', 'seeing_tertiant_1', 'seeing_tertiant_2', 'seeing_tertiant_3', 'seeing_encompassing_all_tertiants', 'seeing_defabricate_understanding']
    #   valence_obs_names: ['noticing_v1', 'noticing_v2']
    
    # Various V2 actions which can be taken
    action_names = ['inspect_tertiant_1', 'inspect_tertiant_2', 'inspect_tertiant_3', 'choose_face_happy', 'choose_face_sad']
    if action=='choose_face_happy':
      # if self.verbose: print(f'L2 get_obs: Producing sampling observation...')
      if self.tertiants[self.L2_timestep][0]=='eye_left' and self.tertiants[self.L2_timestep][1]=='mouth_up' and self.tertiants[self.L2_timestep][2]=='eye_right':
        what_obs='correct'
      else:
        what_obs='incorrect'
      where_obs = 'seeing_face_happy_understanding'
    elif action=='choose_face_sad':
      # if self.verbose: print(f'L2 get_obs: Producing sampling observation...')
      if self.tertiants[self.L2_timestep][0]=='eye_left' and self.tertiants[self.L2_timestep][1]=='mouth_down' and self.tertiants[self.L2_timestep][2]=='eye_right':
        what_obs = 'correct'
      else:
        what_obs = 'incorrect'
      where_obs = 'seeing_face_sad_understanding'
    elif action=='inspect_tertiant_1':
      if self.tertiants[self.L2_timestep][0] == 'mouth_up':
        what_obs = 'seeing_mouth_up'
      elif self.tertiants[self.L2_timestep][0] == 'mouth_down':
        what_obs = 'seeing_mouth_down'
      elif self.tertiants[self.L2_timestep][0] == 'eye_left':
        what_obs = 'seeing_eye_left'
      elif self.tertiants[self.L2_timestep][0] == 'eye_right':
        what_obs = 'seeing_eye_right'
      where_obs = 'seeing_tertiant_1'
    elif action=='inspect_tertiant_2':
      if self.tertiants[self.L2_timestep][1] == 'mouth_up':
        what_obs = 'seeing_mouth_up'
      elif self.tertiants[self.L2_timestep][1] == 'mouth_down':
        what_obs = 'seeing_mouth_down'
      elif self.tertiants[self.L2_timestep][1] == 'eye_left':
        what_obs = 'seeing_eye_left'
      elif self.tertiants[self.L2_timestep][1] == 'eye_right':
        what_obs = 'seeing_eye_right'
      where_obs = 'seeing_tertiant_2'
    elif action=='inspect_tertiant_3':
      if self.tertiants[self.L2_timestep][2] == 'mouth_up':
        what_obs = 'seeing_mouth_up'
      elif self.tertiants[self.L2_timestep][2] == 'mouth_down':
        what_obs = 'seeing_mouth_down'
      elif self.tertiants[self.L2_timestep][2] == 'eye_left':
        what_obs = 'seeing_eye_left'
      elif self.tertiants[self.L2_timestep][2] == 'eye_right':
        what_obs = 'seeing_eye_right'
      where_obs = 'seeing_tertiant_3'
    elif action=='choose_defabricate':
      what_obs = 'seeing_defabricate'
      where_obs = 'seeing_defabricate_understanding'
    else:
      what_obs = 'null'
      where_obs='seeing_eyelids'
    valence_obs = 'noticing_v2'
    return what_obs, where_obs, valence_obs
  
  def get_current_location(self): # Returns current location (to be sent to L1 gen process)
    return self.current_location
  
  def get_tertiant_contents(self):
    tertiant1=self.tertiants[self.L2_timestep][0]
    tertiant2=self.tertiants[self.L2_timestep][1]
    tertiant3=self.tertiants[self.L2_timestep][2]
    return [tertiant1, tertiant2, tertiant3]
  
  # Fetch L2 timestep, so that L1 gen process can gain access to it
  def get_L2_timestep(self):
    return self.L2_timestep
  
  def defabricate_A(self, A, dimensions, beliefs):
    print('defabricate_A')
    Mode = 'beliefs_softmax' # ['empty', 'beliefs_softmax', 'beliefs_and_A_softmax']
    if Mode == 'empty':
      A_defab = utils.initialize_empty_A(dimensions['num_obs'], dimensions['num_states'])
      return A_defab, beliefs
    elif Mode == 'beliefs_softmax':
      beliefs_0=np.array(beliefs[0])
      beliefs_1=np.array(beliefs[1])
      beliefs[0] = maths2.softmax_precision(beliefs_0, 0.7)
      beliefs[1] = maths2.softmax_precision(beliefs_1, 0.7)
      A_defab_0=A[0]
      A_defab_1=A[1]
      return [A_defab_0, A_defab_1], beliefs
    elif Mode == 'beliefs_and_A_softmax': # use if we want to defabricate A matrix beliefs/params as well
      beliefs_0=np.array(beliefs[0])
      beliefs_1=np.array(beliefs[1])
      beliefs[0] = maths2.softmax_precision(beliefs_0, 0.7)
      beliefs[1] = maths2.softmax_precision(beliefs_1, 0.7)
      A_defab_0=maths2.softmax_precision(A[0], 0.9)
      A_defab_1=maths2.softmax_precision(A[1], 0.9)
      return [A_defab_0, A_defab_1], beliefs
    else:
      print('Error in L2 env: defabricate_A function!')
    return
  
  def update_context(self, received_where_obs_dist_L3):
    if received_where_obs_dist_L3[1]==1: self.context='defabricate'
    elif received_where_obs_dist_L3[0]==1: self.context='normal'
    context = self.context
    return context
  
class FacialFeatures(Env):
  def __init__(self, precision=1.0, facial_feature_state = None, sampling_state = None, tertiant_contents=None, where=None, mappings=None, dimensions=None, verbose=False, L2_timestep=None): # start with eyes closed by default
    
    self.facial_feature_state_names = mappings['facial_feature_state_names']
    self.facial_feature_obs_names = mappings['facial_feature_obs_names']
    self.sampling_obs_names = mappings['sampling_obs_names']
    self.sampling_action_names = mappings['sampling_action_names']
    self.valence_obs_names = mappings['valence_obs_names']
    self.L2_timestep = L2_timestep
    self.tertiants=tertiant_contents

    # amount of noise in the env
    self.precision=precision # set precision
    self.verbose=verbose
    self.L1_timestep=0
    self.where = where
    if verbose: print(f'V1: self.where: [{self.where} at t_L1: [{self.L1_timestep}]]')
    
    # Set sampling_state:
    if sampling_state is None:
      self.sampling_state = 'sampling'
    else:
      self.sampling_state=sampling_state

    # Set valence_state:
    self.valence_state = 'good' # set the default valence observation to v1, that is the best possible valence that is always produced in tandem with a L1 observation

    # Set facial_feature_state:
    # what features to display in each tertiant at each timestep (max 23 - in a single sampling session?)
    # perhaps put all three into an array in the future for unlimited locations?
    #
    # The stimuli cannot change in L1 timesteps

    if self.where == 'tertiant_1':
      if self.tertiants[self.L2_timestep][0] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][0] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][0] == 'eye_left':
        self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
      elif self.tertiants[self.L2_timestep][0] == 'eye_right':
        self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    elif self.where == 'tertiant_2':
      if self.tertiants[self.L2_timestep][1] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][1] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][1] == 'eye_left':
        self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
      elif self.tertiants[self.L2_timestep][1] == 'eye_right':
        self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    elif self.where == 'tertiant_3':
      if self.tertiants[self.L2_timestep][2] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][2] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][2] == 'eye_left':
        self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
      elif self.tertiants[self.L2_timestep][2] == 'eye_right':
        self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    else:
      self.object_presentation=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # null - i.e. there is no location being looked at so produce 'null'
    object_pres=self.object_presentation
    if self.where == None: #Default to 'null' location
      self.where == 'null'

    # Load initial states based on tertiant being inspected
    self.facial_feature_state = self.facial_feature_state_names[self.object_presentation[0]]
    
    if self.verbose: print(f'V1 init: facial feature state: {self.facial_feature_state}')  
    if self.verbose: print(f'V1 init: sampling state: {self.sampling_state}')  
       
    # default observations upon initialization of L1 env
    self.facial_feature_obs=self.get_observation('go_sample')[0]
    self.sampling_obs=self.get_observation('go_sample')[1]

    #print("V1 init: sampling_states:", sampling_states)
    self.num_states = dimensions['num_states'] # load this global variable into the Env V1
    self.facial_feature_dist=self.generate_feature_dist()
    
  #Take a step through generative process, t=t+1
  def step(self, action):
    #self.sampling_states(action) # the only action is 'sampling'
    if self.verbose: print(f'FacialFeaturesEnv step input: {action}')
    self.L1_timestep = self.L1_timestep+1
    action_choice = action
    if action == 'go_sample':
      self.sampling_state='sampling'
    elif action == 'stop_sampling':
      self.sampling_state='stopping'
    else:
      print('Error in V1 step!')
    self.change_tertiant(where=self.where)
    return self.get_observation(action=action_choice)

  # Give initial observations for a given location... set the conditions to not return anything useful unless there is a location and sampling is active
  def reset(self, where=None, action=None, L2_timestep=None):
    if action=='go_sample':
      self.sampling_state='sampling'
    elif action=='stop_sampling':
      self.sampling_state='stopping'
    else:
      print('Error in V1 reset: invalid sampling action!')
    self.valence_state = 'good'
    self.L1_timestep = 0
    self.L2_timestep = L2_timestep
    self.action = action
    self.where=where # what is the current object of the gaze
    if self.verbose: print(f'env_L1 reset: where: [{where}]')
    if self.verbose: print(f'env_L1 reset: action: [{action}]')
    if self.where == 'tertiant_1':
      if self.tertiants[self.L2_timestep][0] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][0] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][0] == 'eye_left':
        self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
      elif self.tertiants[self.L2_timestep][0] == 'eye_right':
        self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    elif self.where == 'tertiant_2':
      if self.tertiants[self.L2_timestep][1] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][1] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][1] == 'eye_left':
        self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
      elif self.tertiants[self.L2_timestep][1] == 'eye_right':
        self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    elif self.where == 'tertiant_3':
      if self.tertiants[self.L2_timestep][2] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][2] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth up / 4 would be mouth down
      elif self.tertiants[self.L2_timestep][2] == 'eye_left':
        self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
      elif self.tertiants[self.L2_timestep][2] == 'eye_right':
        self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    else:
      self.object_presentation=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # null - i.e. there is no location being looked at so produce 'null'
    if self.verbose: print(f'key: timestep {self.L2_timestep}: self.tertiants[self.L2_timestep][1]: {self.tertiants[self.L2_timestep][1]}')
    self.change_tertiant(where=self.where)
    output = self.get_observation(action=self.action)
    if self.verbose: print(f'resetting V1 states to:\tfeature: {output[0]}\tsampling_state: {output[1]}')
    return output
  
  # This will either return '1 0 0 0 0' if there is no feature, or it will return a softmaxed distribution
  # with self.facial_feature as the dominant value

  def generate_feature_dist(self):
    # Load input feature into stateidx
    stateidx = self.facial_feature_state_names.index(self.facial_feature_state)
    
    # If no facial feature has been set then choose
    if self.facial_feature_state == 'null':
        self.facial_feature_dist_nosoft = utils.onehot(stateidx, self.num_states[0])
        self.facial_feature_dist = self.facial_feature_dist_nosoft
        #if self.verbose: print(f'V1 Gen Feature Dist (null case): self.facial_feature_dist: {self.facial_feature_dist}')
    else:
        self.facial_feature_dist_nosoft = np.zeros(self.num_states[0])
        self.facial_feature_dist = np.zeros(self.num_states[0])
        
        # How do I create the distribution for 5*2 instead of 5*1?
        self.facial_feature_dist_nosoft[1:] = self.precision * utils.onehot(stateidx-1, len(self.facial_feature_state_names)-1)
        self.facial_feature_dist[1:] = self.facial_feature_dist_nosoft[1:]
        if self.verbose: print(f'V1 Gen Feature Dist: self.facial_feature_dist at tertiant: {self.where}: {self.facial_feature_dist}')
    return self.facial_feature_dist
  
  # Generate facial observation for 
  # Input: self.facial_feature_dist 
  # Output: facial_obs
  #
  # We could also do this by having an Env time counter called self.t_env or similar instead of using del
  def get_observation(self, action):
    # generate the observations for each of the three tertiants at the given timestep
    if self.verbose: print(f'\nV1: object at time t={self.L1_timestep}: [{self.object_presentation[self.L1_timestep]}]')
    if action=='go_sample':
      self.facial_feature_obs=self.facial_feature_obs_names[self.object_presentation[self.L1_timestep]]
      self.sampling_obs=self.sampling_obs_names[0] # proprioceptive feedback that the agent has just sampled/is sampling
      if self.verbose: print(f'V1 get_obs: Produced obs {self.facial_feature_obs} : {self.sampling_obs}')
    elif action=='stop_sampling':
      self.facial_feature_obs=self.facial_feature_obs_names[0] # return null observation if decided not to sample
      self.sampling_obs=self.sampling_obs_names[1] # proprioceptive feedback that agent has stopped sampling
      if self.verbose: print(f'V1 get_obs: observation provided is {self.facial_feature_obs}, {self.sampling_obs}')
    else:
      self.facial_feature_obs = 'null'
      self.sampling_obs='noticed_stopping'
      if self.verbose: print(f"Error in V1 Get_observation!")
    if self.valence_state == 'good': # this is currently the only L1 valence observation possible
      self.valence_obs = 'noticing_v1'
    else:
      self.valence_obs = 'null'
    return self.facial_feature_obs, self.sampling_obs, self.valence_obs

  def change_tertiant(self, where='eyelids'): # default to eyelids location
    # what features to display in each tertiant at each timestep (max 23 - in a single sampling session?)
    # perhaps put all three into an array in the future for unlimited locations?
    if self.verbose: print(f'FacialFeatureEnv: change_tertiant: changing location to: {where}')
    if where == 'tertiant_1':
      self.object_presentation=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 1 is left eye
    elif where == 'tertiant_2':
      if self.tertiants[self.L2_timestep][1] == 'mouth_up':
        self.object_presentation=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # 3 is mouth up
      elif self.tertiants[self.L2_timestep][1] == 'mouth_down':
        self.object_presentation=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] # 3 is mouth down
    elif where == 'tertiant_3':
      self.object_presentation=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 2 is right eye
    else:
      self.object_presentation=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # null - i.e. there is no location being looked at so produce 'null'
    if self.verbose: print(f'FacialFeatureEnv: change_tertiant: Object Presentation: {self.object_presentation[0]}')
    return
  
  def get_sampling_state(self):
    if self.verbose: print(f'Getting current sampling_state for V1')
    sampling_state = self.sampling_state
    return sampling_state