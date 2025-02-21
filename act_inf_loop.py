# Active Inference Loop for Meditative Deconstruction via Letting Go Model
# Shawn Prest, modified and extended from Heins et al. Deep Scene Construction model

#from pymdp.agent import Agent
from agent_env_writable import Agent
from pymdp.envs import Env
from pymdp import utils, maths, control, learning
import numpy as np
from copy import deepcopy
import leddy_L3_agent_init as init_L3
import leddy_L2_agent_init as init_L2
import leddy_L1_agent_init as init_L1
import leddy_Env as leddy_env
import hier_link_utils as link
import plotter
import maths_custom as maths2
import csv # for importing stimuli lists & for writing belief data at each timestep to file for later graphing
import os
import os.path

verbose = True

np.set_printoptions(formatter={'float': '{:.2f}'.format})

# L2 History
beliefs_hist_L2=[]
policy_beliefs_hist_L2=[]
efe_beliefs_hist_L2=[]
action_hist_L2=[]

#L1 History
beliefs_hist_L1=[]
policy_beliefs_hist_L1=[]
efe_beliefs_hist_L1=[]
action_hist_L1=[]
link_hist_L1toL2=[]

# Env Hist
env_obs_hist_L2=[]
env_obs_hist_L1=[]
env_locations_hist_L2=[]

##---------------------------------------------------------------
# Config

# L3 preference to not feel bad:
pref_to_not_feel_bad = -2 # Set to negative to motivate defabrication (-2), zero for no motivation to defabricate
# L2 reward for correct answer:
reward_for_correct=8 # 8, 10 works well, 9
reward_for_incorrect=-5 # -0.5, -0.3 works well - previous 2 - most recent previous = -0.3

# L2 reward for defabricating
reward_for_defabricate=-15 # 4.5 results in 4 non-consecutive defabrications

# L2 valence preference to not feel 'bad'
reward_for_bad=-0
reward_for_good=0

# L1 reward if agent stops sampling:
reward_for_stopping = 0.0001

# Threshold for the point at which sampling at L1 stops
confidence_threshold=0.90

T_L3=11
T_L2=5 # number of trials - previous: 80
T_L1=5 # max L1 samples

t_L3=0
t_L2=0 # go from 0-n (n+1 steps)
t_L1=0
t_L32=0 # this counts all of the t2 timesteps which occur over all t_L3

display_history_L2=True
display_history_L1=True

# End of config
#---------------------------------------------------------------

# Define L2 Gen Model
parameters_L3, mappings_L3, dimensions_L3 = init_L3.init_L3_generative_model(pref_to_not_feel_bad=pref_to_not_feel_bad, verbose=False)
my_agent_L3 = Agent(A=parameters_L3['A'], B=parameters_L3['B'], C=parameters_L3['C'], D=parameters_L3['D'])
parameters_L2, mappings_L2, dimensions_L2 = init_L2.init_V2_generative_model(reward_for_correct=reward_for_correct, reward_for_incorrect = reward_for_incorrect, reward_for_defabricate=reward_for_defabricate, reward_for_bad=reward_for_bad, reward_for_good=reward_for_good, verbose=False)
my_agent_L2 = Agent(A=parameters_L2['A'], B=parameters_L2['B'], C=parameters_L2['C'], D=parameters_L2['D'], pA=parameters_L2['pA'], use_param_info_gain=True, policy_len=1, lr_pA=0.0, gamma=16.0) #lr_PA was 0.6

# Fire up L2 Env Gen Process

# Load stimuli file...
if os.path.isfile('input_stimuli.csv'): #check if the stimuli file exists
    print(f'Loading stimuli from input_stimuli.csv...')
    stimuli_file = open('input_stimuli.csv')
    csvreader = csv.reader(stimuli_file)
    tertiant_contents_from_file = []
    tertiant_contents_from_file = list(csvreader)
    stimuli_file.close()
    tertiant_contents = tertiant_contents_from_file # old hard coded: ['eye_left', 'mouth_up', 'eye_right']
    print(f'Tertiant Contents ***: {tertiant_contents[0]}')
else:
    print(f'Loading stimuli from internal array: eye_left, mouth_up, eye_right...')
    tertiant_contents=['eye_left', 'mouth_up', 'eye_right']

my_env_L3 = leddy_env.ValenceContext(starting_context = 'normal', starting_location='normal', mappings=mappings_L3, verbose=False)
received_where_obs_dist_L3=utils.onehot(0, len(mappings_L3['where_obs_names'])) # set this vector initially to normal, could do this in some other way, getting it from my_env_L3 above

tertiant_location = 'eyelids' # starting location for L2
my_env_L2 = leddy_env.Faces(starting_location=tertiant_location, tertiant_contents = tertiant_contents, mappings=mappings_L2, verbose=False)
received_what_obs_label_L2, received_where_obs_label_L2, received_valence_obs_label_L2 = my_env_L2.reset()
where_state_label_L2 = my_env_L2.get_current_location()
obs_idx_L2=[mappings_L2['what_obs_names'].index(received_what_obs_label_L2), mappings_L2['where_obs_names'].index(received_where_obs_label_L2), mappings_L2['valence_obs_names'].index(received_valence_obs_label_L2)]

# Define L1 Gen Model
parameters_L1, mappings_L1, dimensions_L1 = init_L1.init_V1_generative_model(verbose=False, reward_for_stopping=reward_for_stopping)
my_agent_L1 = Agent(A=parameters_L1['A'], pA=parameters_L1['pA'], B=parameters_L1['B'], C=parameters_L1['C'], D=parameters_L1['D'], use_param_info_gain=True, lr_pA = .3) # previous .5

# Fire up L1 Env Gen Process
tertiant_contents_L1=my_env_L2.get_tertiant_contents()
my_env_L1 = leddy_env.FacialFeatures(precision=1.0, sampling_state = 'sampling', where=tertiant_location, tertiant_contents=tertiant_contents, mappings=mappings_L1, dimensions=dimensions_L1, verbose=False, L2_timestep=my_env_L2.get_L2_timestep())
facial_feature_obs_label_L1, sampling_obs_label_L1, valence_obs_label_L1 = my_env_L1.reset(action='go_sample', L2_timestep=my_env_L2.get_L2_timestep())
obs_idx_L1 = [mappings_L1['facial_feature_obs_names'].index(facial_feature_obs_label_L1), mappings_L1['sampling_obs_names'].index(sampling_obs_label_L1), mappings_L1['valence_obs_names'].index(valence_obs_label_L1)]

# Get initial priors from L3 agent for initial L2 agent time step prior to empirical prior
prior_L3 = my_agent_L3.D

# Get initial priors from L2 agent for initial L1 agent time step prior to empirical prior
prior_L2 = my_agent_L2.D

# Store initial belief
valence_beliefs_hist_L3 = np.zeros((T_L2*T_L3, dimensions_L3['num_states'][0]))

#L2 Beliefs for Plotting # [int((str(t_L3)+str(t_L2)))
face_beliefs_hist_L2 = np.zeros((T_L2*T_L3, dimensions_L2['num_states'][0]))
valence_beliefs_hist_L2 = np.zeros((T_L2*T_L3, dimensions_L2['num_states'][2]))
search_choices_hist_L2 = []

#L1 Beliefs for Plotting
feature_beliefs_hist_L1 = np.zeros((T_L3, T_L2, T_L1, dimensions_L1['num_states'][0]))
sampling_prob_hist_L1 = np.zeros((T_L2, T_L1, 2))

for t_L3 in range(0, T_L3): # Level 3 loop - Valence Context    
    my_agent_L3.reset()
    where_prior_from_L3 = link.get_valence_context_prior_from_L3(beliefs_high=prior_L3, A_above=my_agent_L3.A, dimensions_low=dimensions_L2, verbose=True) # Get valence context prior from L3
    my_agent_L2.set_qs(qs=where_prior_from_L3, state_factor='2')
    L3_context=my_env_L2.update_context(received_where_obs_dist_L3)
    for t_L2 in range(0, T_L2): # Level 2 loop - Face Type
        valence_beliefs_hist_L3[t_L32, :] = prior_L3[0].copy() # For plotting
        env_obs_hist_L2.append((t_L2, obs_idx_L2)) # save Env Obs hist
        # Send L2 prior -> L1 
        prior_from_above = link.get_prior_from_above_classic(beliefs_high=prior_L2, A_above=my_agent_L2.A, dimensions_low=dimensions_L1, verbose=False)
        posterior_L1 = prior_from_above # sets the initial limit on sampling at L1
        my_agent_L1.reset(init_qs=prior_from_above)
        
        # Sample in the current L2 location
        t_L1=0
        # update L1 env to appropriate where state and action
        where_state_label_L2 = my_env_L2.get_current_location() # initially look here, it is 'eyelids' initially 
        env_locations_hist_L2.append((t_L2, where_state_label_L2))
      
        if (where_state_label_L2 == 'tertiant_1') or (where_state_label_L2 == 'tertiant_2') or (where_state_label_L2 == 'tertiant_3'):
            sampling_action = 'go_sample'    
            facial_feature_obs_label_L1, sampling_obs_label_L1, valence_obs_label_L1 = my_env_L1.reset(where = where_state_label_L2, action = sampling_action, L2_timestep=my_env_L2.get_L2_timestep())
            sampling_state=my_env_L1.get_sampling_state()
            env_obs_hist_L1.append((t_L2, t_L1, facial_feature_obs_label_L1, sampling_obs_label_L1))

            # L1 Loop
            while (t_L1 < T_L1) and (posterior_L1[0][np.argmax(posterior_L1[0])]<=confidence_threshold):
                posterior_L1 = my_agent_L1.infer_states(obs_idx_L1)
                beliefs_hist_L1.append((t_L2, t_L1, posterior_L1[0].copy(), posterior_L1[1].copy(), posterior_L1[2].copy())) # record timestep L2 L1 and three state factor posterior beliefs (facial_feat_state, sampling_state, valence_state)
                feature_beliefs_hist_L1[t_L3, t_L2, t_L1, :]=posterior_L1[0].copy() # For plotting
                policy_posterior_L1, efe_L1 = my_agent_L1.infer_policies()
                policy_beliefs_hist_L1.append((t_L2, t_L1, policy_posterior_L1.copy()))
                chosen_action_id_L1 = my_agent_L1.sample_action()
                action_hist_L1.append((t_L2, t_L1, mappings_L1['sampling_action_names'][int(chosen_action_id_L1[1])]))
                facial_feature_obs_label_L1, sampling_obs_label_L1, valence_obs_label_L1 = my_env_L1.step(mappings_L1['sampling_action_names'][int(chosen_action_id_L1[1])])
                sampling_state = my_env_L1.get_sampling_state()
                obs_idx_L1 = mappings_L1['facial_feature_obs_names'].index(facial_feature_obs_label_L1), mappings_L1['sampling_obs_names'].index(sampling_obs_label_L1), mappings_L1['valence_obs_names'].index(valence_obs_label_L1)
                t_L1 += 1
                # End of L1 Loop
        elif where_state_label_L2 == 'eyelids': # agent is either seeing eyelids and is not going to sample or has a mental understanding and is not going to sample
            sampling_action = 'stop_sampling'
        else:
            if verbose: print("L1 Loop skipped b/c understanding")
        
        # send messages L1->L2
        if (received_where_obs_label_L2 == 'seeing_tertiant_1') or (received_where_obs_label_L2 == 'seeing_tertiant_2') or (received_where_obs_label_L2 == 'seeing_tertiant_3'):
            obs_dist_L2 = link.get_obs_from_below(beliefs_from_below=posterior_L1, where_obs_from_above=received_where_obs_label_L2, valence_obs_from_below=received_valence_obs_label_L2, dimensions_from_above=dimensions_L2, mappings_from_above=mappings_L2, verbose=False)
            obs_dist_L2[1]=utils.onehot(mappings_L2['where_obs_names'].index(received_where_obs_label_L2), dimensions_L2['num_obs'][1])
            use_dist=True
        else:
            # Create the obs_dist_L2 data structure
            obs_dist_L2 = utils.obj_array_zeros(dimensions_L2['num_obs'])
                        
            # create distributions for expected L2 observations based on previous observations
            obs_dist_L2[0]=utils.onehot(mappings_L2['what_obs_names'].index(received_what_obs_label_L2), dimensions_L2['num_obs'][0])
            obs_dist_L2[1]=utils.onehot(mappings_L2['where_obs_names'].index(received_where_obs_label_L2), dimensions_L2['num_obs'][1])
            obs_dist_L2[2]=utils.onehot(mappings_L2['valence_obs_names'].index(received_valence_obs_label_L2), dimensions_L2['num_obs'][2])
            obs_dist_L2[2][:]=[0.06, 0.94] # L2 valence observations are mostly bad ('noticing_v2') # This should be in the generative process as an observation
            use_dist = True
        link_hist_L1toL2.append([t_L2, obs_dist_L2[2]])
        posterior_L2 = my_agent_L2.infer_states(obs_dist_L2, distr_obs = use_dist) # do inference using expected observation distribution to get new posterior beliefs
        face_beliefs_hist_L2[t_L32, :]= posterior_L2[0].copy() # For plotting
        valence_beliefs_hist_L2[t_L32, :] = posterior_L2[2].copy() # For plotting
        policy_posterior_L2, efe_L2 = my_agent_L2.infer_policies()
        policy_beliefs_hist_L2.append((t_L2, policy_posterior_L2, efe_L2))
        chosen_action_id_L2 = my_agent_L2.sample_action()
        beliefs_hist_L2.append([t_L3, t_L2, prior_L3[0], posterior_L2[0].copy(), posterior_L2[1].copy(), posterior_L2[2].copy(), mappings_L2['action_names'][int(chosen_action_id_L2[1])]])
        action_hist_L2.append((t_L3, t_L2, mappings_L2['action_names'][int(chosen_action_id_L2[1])],mappings_L2['valence_action_names'][int(chosen_action_id_L2[2])])) # For plotting
        search_choices_hist_L2.append(mappings_L2['action_names'][int(chosen_action_id_L2[1])]) # For plotting
        
        # Take env_L2 step
        # we should actually have the generative process altering the agent's generative model for prior beliefs and A matrix params, but there is no direct Env->Agent bridge, so we bridge them here
        # if the agent has not chosen to defabricate then the A matrix and beliefs are not altered
        received_what_obs_label_L2, received_where_obs_label_L2, received_valence_obs_label_L2, A_defab_L2, posterior_L2_defab = my_env_L2.step(mappings_L2['action_names'][int(chosen_action_id_L2[1])], mappings_L2['valence_action_names'][int(chosen_action_id_L2[2])], my_agent_L2.A.copy(), dimensions_L2, posterior_L2.copy())
        my_agent_L2.A[0]=A_defab_L2[0]
        my_agent_L2.A[1]=A_defab_L2[1]
        my_agent_L2.set_qs(posterior_L2_defab) # Update the agent's beliefs to the new defabricated ones, i.e. less precision.
        
        # Get expected states over next location, given the L2 action taken by agent
        obs_idx_L2 = mappings_L2['what_obs_names'].index(received_what_obs_label_L2), mappings_L2['where_obs_names'].index(received_where_obs_label_L2), mappings_L2['valence_obs_names'].index(received_valence_obs_label_L2)
        prior_L2=posterior_L2_defab
        my_agent_L2.update_A([mappings_L2['what_obs_names'].index(received_what_obs_label_L2), mappings_L2['where_obs_names'].index(received_where_obs_label_L2),mappings_L2['valence_obs_names'].index(received_valence_obs_label_L2)]) # update A matrices incorporating what, where and valence observations from env not the agent's inferred observations!        
        t_L32=t_L32+1
    
    # L3 processing after level 2 finished---------------------------------------------------
    obs_dist_L3 = link.get_obs_from_L2(beliefs_from_below=posterior_L2_defab, dimensions_from_above=dimensions_L3, mappings_from_above=mappings_L3, verbose=False)
    posterior_L3 = my_agent_L3.infer_states(obs_dist_L3, distr_obs = True)
    policy_posterior_L3, efe_L3 = my_agent_L3.infer_policies()
    chosen_action_id_L3=my_agent_L3.sample_action()
    sampling_action = mappings_L3['where_action_names'][int(chosen_action_id_L3[1])]

    # Step L3
    received_where_obs_dist_L3 = my_env_L3.step(mappings_L3['where_action_names'][int(chosen_action_id_L3[1])])
    prior_L3=posterior_L3

if display_history_L2 == True:
    if verbose: print(f'\nBeliefs over time L2:\ntL3.tL2 norm\tbad\tfH fS\t\t ld   t1   t2   t3   fHu  fSu  dfU\tbad neutral good\taction')
    for moment in beliefs_hist_L2:
        if verbose: print(f'{moment[0]}.{moment[1]}\t{np.round(moment[2], 2)}\t{np.round(moment[3], 2)}\t{np.round(moment[4], 2)}\t{np.round(moment[5], 2)}\t{moment[6]}')

    if verbose: print(f'\nPolicy beliefs over time L2:\nt_L2\tPolicy Beliefs')
    for policy in policy_beliefs_hist_L2:
        if verbose: print(f'{policy[0]}\t{np.round(policy[1], 2)}\t\t\tefe:{np.round(policy[2], 2)}')

    if verbose: print(f'\nPolicy prediction error beliefs over time L2:\nt_L2\tPE')
    for efe in efe_beliefs_hist_L2:
        if verbose: print(f'{efe[0]}  {np.round(efe[1], 3)}')
    
    if verbose: print(f'\nAction choices over time L2:\nt_L2\tACTION CHOICE')
    for action in action_hist_L2:
        print(f'{action[0]}.{action[1]}\t{action[2]}')

    if verbose: print(f'\n\nV2 observation history L2:\nt_L2\twhat\twhere\tvalence')
    for obs in env_obs_hist_L2:
        if verbose: print(f"{obs[0]}\t{mappings_L2['what_obs_names'][obs[1][0]]}\t\t{mappings_L2['where_obs_names'][obs[1][1]]}\t\t{mappings_L2['valence_obs_names'][obs[1][2]]}")

    if verbose: print(f'\n\nV2 location history L2:\nt_L2\t')
    for location in env_locations_hist_L2:
        if verbose: print(f"{location[0]}\t{location[1]}")
    
    # print L1 belief history to csv file:
    L2_beliefs_file = open('L2_beliefs.csv', 'w', newline='')
    writer2=csv.writer(L2_beliefs_file)
    writer2.writerow(['Time L3', 'Time L2', 'Feel Okay', 'Feed Bad', 'Face Happy', 'Face Sad', 'Eyelid', 'Tertiant 1', 'Tertiant 2', 'Tertiant 3', 'Face Happy Understanding', 'Face Sad Understanding', 'Defabrication Understanding'])
    for what in beliefs_hist_L2:
        writer2.writerow([what[0], what[1], np.round(what[2][0], 2), np.round(what[2][1], 2), np.round(what[3][0], 2), np.round(what[3][1], 2), np.round(what[4][0], 2), np.round(what[4][1], 2), np.round(what[4][2], 2), np.round(what[4][3], 2), np.round(what[4][4], 2), np.round(what[4][5], 2), np.round(what[4][6], 2)])
    L2_beliefs_file.close()
if display_history_L1== True:
    if verbose: print(f'\n\nV1 observation history L1:')
    if verbose: print(f"t_L1\tFEAT.\t\t\t\tSAMP.")
    for obs in env_obs_hist_L1:
        if verbose: print(f"{obs[0]}.{obs[1]}\t{obs[2]}\t\t{obs[3]}")

    # print L1 belief history to csv file:
    L1_beliefs_file = open('L1_beliefs.csv', 'w', newline='')
    writer=csv.writer(L1_beliefs_file)
    writer.writerow(['Time L2', 'Time L1', 'Nul', 'Left Eye', 'Right Eye', 'Mouth Up', 'Mouth Down'])

    L2_L1_ratio = T_L2*T_L3 / (len(beliefs_hist_L1))
    L2_L1_ratio_top=(T_L2*T_L3)
    L2_L1_ratio_bottom=(len(beliefs_hist_L1))
    average_valence=1/L2_L1_ratio

    # print L1 belief history here:
    # L1 belief info
    if verbose: print(f'\nBeliefs L1:\ntime\tL2\t\tnul  eL   eR   mU   mD\t\t\tSAMP/NOT\t\t\tVALENCE_GN')
    
    for what in beliefs_hist_L1:
        #print(f'{what[0]}:{what[1]} {what[2]}:{what[3]}')
        if verbose: print(f'{what[0]}.{what[1]}\t{np.round(beliefs_hist_L2[what[0]][3],2)}\t{np.round(what[2], 2)}\t\t{np.round(what[3], 2)}\t\t\t{np.round(what[4], 2)}')
        writer.writerow([what[0], what[1], np.round(what[2][0], 2), np.round(what[2][1], 2), np.round(what[2][2], 2), np.round(what[2][3], 2), np.round(what[2][4], 2)])
    
    writer.writerow(['L2 L1 ratio:', L2_L1_ratio])
    if verbose: print(f'\nL2 L1 ratio (beliefs_hist_L1[-1][0]): {beliefs_hist_L1[-1][0]}')
    if verbose: print(f'\nL2 L1 ratio: {L2_L1_ratio}')
    if verbose: print(f'\nL2 L1 ratio top over bottom: {L2_L1_ratio_top} / {L2_L1_ratio_bottom}')
    if verbose: print(f'\nAverage Affective Valence: {average_valence}')
    L1_beliefs_file.close()

# feature_beliefs_hist_L1[0] - the '0' refers to the first t_L3 timestep
print(beliefs_hist_L1[0][2])
print(beliefs_hist_L1[1][2])
print(beliefs_hist_L1[2][2])
print(beliefs_hist_L1[3][2])

# Transfer from list to array
feature_beliefs_hist_L1[0][0][0]=np.array(beliefs_hist_L1[0][2])
feature_beliefs_hist_L1[0][0][1]=np.array(beliefs_hist_L1[1][2])
feature_beliefs_hist_L1[0][0][2]=np.array(beliefs_hist_L1[2][2])
feature_beliefs_hist_L1[0][1][0]=np.array(beliefs_hist_L1[3][2])
feature_beliefs_hist_L1[0][1][1]=np.array(beliefs_hist_L1[4][2])
feature_beliefs_hist_L1[0][1][2]=np.array(beliefs_hist_L1[5][2])
feature_beliefs_hist_L1[0][1][3]=np.array(beliefs_hist_L1[6][2])
feature_beliefs_hist_L1[0][2][0]=np.array(beliefs_hist_L1[7][2])
print(feature_beliefs_hist_L1[0][2][0])

plotter.plot_beliefs_hierarchical(feature_beliefs_hist_L1[0], env_locations_hist_L2, valence_beliefs_hist_L3, face_beliefs_hist_L2, valence_beliefs_hist_L2, action_hist_L2, T_L2, [0, 1, 2, 3], mappings_L2)
plotter.plt.show(block=True)