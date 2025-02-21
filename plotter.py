from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns


def plot_beliefs_hierarchical(beliefs_low, env_locations_hist_L2, valence_beliefs_L3, beliefs_high, valence_beliefs_high, action_hist_L2, T_L2, t_high_to_show, mappings_high=None):
    ## beliefs_hist_L2.append([t_L3, t_L2, prior_L3[0], posterior_L2[0].copy(), posterior_L2[1].copy(), posterior_L2[2].copy(), mappings_L2['action_names'][int(chosen_action_id_L2[1])]])
    
    fig = plt.figure(tight_layout=False,figsize=(10,12))

    num_sub_t_step = len(t_high_to_show)
    
    gs = gridspec.GridSpec(6, 3)

    T_high = len(beliefs_high)
    ax_topV3 = fig.add_subplot(gs[0, :])
    ax_top = fig.add_subplot(gs[1, :])
    ax_topV = fig.add_subplot(gs[2, :])
    #ax_topT = fig.add_subplot(gs[3, :])
    imdataV3 = ax_topV3.imshow(valence_beliefs_L3.T, clim = (0.0, 1.0))
    imdata = ax_top.imshow(beliefs_high.T, clim = (0.0, 1.0))
    imdataV = ax_topV.imshow(valence_beliefs_high.T, clim = (0.0, 1.0)) 

    # For L3 context valence
    ax_topV3.set_xticks(np.arange(T_high))
    ax_topV3.set_yticks(np.arange(2))
    ax_topV3.set_yticklabels(labels = ['Okay', 'Bad'], rotation=45) # mappings_high['what_state_names']
    ax_topV3.set_ylabel('L3 Valence belief')
    ax_topV3.set_xlabel('Timestep')
    ax_topV3.set_title('Level 3: Valence Context', fontsize = 10)
    fig.colorbar(imdataV3, ax=ax_topV3)
    
    
    # Face beliefs L2
    ax_top.set_xticks(np.arange(T_high))
    ax_top.set_yticks(np.arange(2))
    ax_top.set_yticklabels(labels = ['Happy Face', 'Sad Face'], rotation=45) # mappings_high['what_state_names']
    ax_top.set_ylabel('Face type belief')
    ax_top.set_xlabel('Timestep')
    ax_top.set_title('Level 2: Faces', fontsize = 10)
    
    fig.colorbar(imdata, ax=ax_top)

    # Overlaying emojis
    emojis = [['😀' for _ in range(54)] for _ in range(2)]  # Example emojis for each block

    # Annotate each cell with an emoji
    
    for i in range(2):
        for j in range(54):
            if action_hist_L2[j][2] == 'inspect_tertiant_1':
                text = ax_top.text(j, i, '1', ha='center', va='center', color='white')
            elif action_hist_L2[j][2] == 'inspect_tertiant_2':
                text = ax_top.text(j, i, '2', ha='center', va='center', color='white')
            elif action_hist_L2[j][2] == 'inspect_tertiant_3':
                text = ax_top.text(j, i, '3', ha='center', va='center', color='white')
            elif action_hist_L2[j][2] == 'choose_face_happy':
                text = ax_top.text(j, i, '😀', ha='center', va='center', color='white')
            elif action_hist_L2[j][2] == 'choose_face_sad':
                text = ax_top.text(j, i, '\U0001F622', ha='center', va='center', color='white')
            else:
                print("Error!")
            #text = ax_top.text(j, i, emojis[i][j], ha='center', va='center', color='white')
    


    # For valence
    ax_topV.set_xticks(np.arange(T_high))
    ax_topV.set_yticks(np.arange(3))
    ax_topV.set_yticklabels(labels = ['Bad', 'Okay', 'Good'], rotation=45) # mappings_high['what_state_names']
    ax_topV.set_ylabel('Valence belief')
    ax_topV.set_xlabel('Timestep')
    ax_topV.set_title('Level 2: Valence', fontsize = 10)
    fig.colorbar(imdataV, ax=ax_topV)

    row = 4 # was 3
    col=0
    t_h=0
    
    while(col<3):
        #if t_h < len(t_high_to_show): # last term was T_L2
        ax = fig.add_subplot(gs[row, col])
        #  These are the lower level facial features
        ax.plot(beliefs_low[t_h,:,0], label = '$P(Null)$')
        ax.plot(beliefs_low[t_h,:,1], label = '$P(Left Eye)$')
        ax.plot(beliefs_low[t_h,:,2], label = '$P(Right Eye)$')
        ax.plot(beliefs_low[t_h,:,3], label = '$P(Upward Mouth)$')
        ax.plot(beliefs_low[t_h,:,4], label = '$P(Downward Mouth)$')
        ax.set_xlim(0, beliefs_low.shape[1])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Feature belief')
        ax.legend(fontsize=8)
        #{np.round(what[3], 2)}
        #ax.set_title('$T_h = $' + f'{t_h}' + f'  {env_locations_hist_L2[t_h][1]}', fontsize=10)
        ax.set_title('t_L2 =' + f'  {col}', fontsize=10)
        col=col+1
        t_h=t_h+1           
        #else:
        #    break
        
def plot_beliefs_subplots(A, title=""):
    """
    Modified from pymdp to use subplots instead of individual plots...
    Utility function that shows a heatmap of a 2-D likelihood (hidden causes in the columns, observations in the rows),
    with hotter colors indicating higher probability.
    """
    row=0
    col=0
    j=0
    fig = plt.figure(tight_layout=False,figsize=(10,12))
    gs = gridspec.GridSpec(4, 4)

    
    while (row<4) and (j<len(A[0])):
        while (col<3) and (j<len(A[0][0][0])):
            ax = fig.add_subplot(gs[row, col])
            ax = sns.heatmap(A[0][:,:,j], cmap="OrRd", linewidth=2.5)
            plt.xticks(range(A[0][:,:,j].shape[1]+1))
            plt.yticks(range(A[0][:,:,j].shape[0]+1))
            #ax.set_ylabel('Probability')
            #ax.set_xlabel('Probability')
            plt.title(title)
            col=col+1
            j=j+1
        row=row+1
        col=0

def plot_beliefs_hierarchical_wL3(beliefs_low, env_locations_hist_L2, beliefs_high, beliefs_hist_L2, dimensions_L2, T_L2, t_high_to_show, mappings_high=None):
    ## beliefs_hist_L2.append([t_L3, t_L2, prior_L3[0], posterior_L2[0].copy(), posterior_L2[1].copy(), posterior_L2[2].copy(), mappings_L2['action_names'][int(chosen_action_id_L2[1])]])
    
    fig = plt.figure(tight_layout=False,figsize=(10,12))

    num_sub_t_step = len(t_high_to_show)
    
    gs = gridspec.GridSpec(4, 3)
    
    T_high = len(beliefs_hist_L2)
    face_beliefs_hist_L2 = np.zeros((T_high, dimensions_L2['num_states'][0]))
    h=0
     #face_beliefs_hist_L2[t_L2, :]= posterior_L2[0].copy() # For plotting
    for belief in beliefs_hist_L2:
        face_beliefs_hist_L2[h, :]=belief[3] #.append([(str(belief[0])+str(belief[1])), belief[3]])
        h+h+1
        
    #beliefs_high = beliefs_high.reshape(beliefs_high.shape[0], 4, 12).sum(2) # average across configurations
    ax_top = fig.add_subplot(gs[0, :])
    imdata = ax_top.imshow(face_beliefs_hist_L2.T, clim = (0.0, 1.0)) # not beliefs_high.T
    #imdata = ax_top.imshow(beliefs_high.T, clim = (0.0, 1.0)) # not beliefs_high.T
    ax_top.set_xticks(np.arange(T_high))
    ax_top.set_yticks(np.arange(2))
    ax_top.set_yticklabels(labels = ['Happy Face', 'Sad Face'], rotation=45) # mappings_high['what_state_names']
    ax_top.set_ylabel('Face type belief')
    ax_top.set_xlabel('Timestep')
    ax_top.set_title('Level 2: Faces', fontsize = 10)
    fig.colorbar(imdata, ax=ax_top)
    
    for t_h in t_high_to_show:
        ax_top.add_patch(Rectangle((t_h-0.5, -0.5), 1, 5, edgecolor='red', fill=False, lw=3))
        #ax_top.text(t_h-0.25, 2.0, f't = {t_h}', fontsize=12.5, color = 'white', rotation = 30)

    row = 1
    col=0
    t_h=0
    
    while(row<5) and (t_h<(len(t_high_to_show))):
        while(col<3):
            if t_h < T_L2:
                #print(f'gs[row, col]: {row},{col}\tsp:{t_h}')
                ax = fig.add_subplot(gs[row, col])
                #  These are the lower level facial features
                ax.plot(beliefs_low[t_h,:,0], label = '$P(Null)$')
                ax.plot(beliefs_low[t_h,:,1], label = '$P(Left Eye)$')
                ax.plot(beliefs_low[t_h,:,2], label = '$P(Right Eye)$')
                ax.plot(beliefs_low[t_h,:,3], label = '$P(Upward Mouth)$')
                ax.plot(beliefs_low[t_h,:,4], label = '$P(Downward Mouth)$')
                ax.set_xlim(0, beliefs_low.shape[1])
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('Feature belief')
                ax.legend(fontsize=8)
                #{np.round(what[3], 2)}
                ax.set_title('$T_h = $' + f'{t_h}' + f'  {env_locations_hist_L2[t_h][1]}', fontsize=10)
                col=col+1
                t_h=t_h+1           
            else:
                break
        if t_h >= T_L2:
            break
        row=row+1
        col=0