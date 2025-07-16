import os
import numpy as np
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.debug:
    import debugpy
    debugpy.listen(5678)
    print('waiting for client to attach')
    debugpy.wait_for_client()  # blocks execution until client is attached

data_folder = Path('./data')

train_folder = os.path.join(data_folder, 'single-task_lift-red-block')

# capire a cosa corrisponde un .npz
# capire meccanismo collezionamento traiettorie robosuite
# conversione da npz/npy to rlds

for ep in sorted(os.listdir(train_folder)):
    all_ep_pieces = sorted([x for x in os.listdir(os.path.join(train_folder, ep)) if '.npz' in x])
    for piece in all_ep_pieces: # every episode is splitted in multiple fragments
        frag_episode = np.load(os.path.join(train_folder, ep, piece), allow_pickle=True)
        # frag_episode.files ['states', 'action_infos', 'successful', 'env']
        # state, actions, success, env = frag_episode['states'], frag_episode['action_infos'] , frag_episode['successful'], frag_episode['env']
        state, gripper_state, image_obs, actions, success, env = \
            frag_episode['states'], frag_episode['gripper_state'], frag_episode['image_obs'], frag_episode['action_infos'] , frag_episode['successful'], frag_episode['env']
        #NOTE: puoi prendere anche lo stato del gripper all'istante (t) come azione per l'immagine all'istante successivo (t+1)
        with open('actions.txt', 'a') as f:
            print(actions, file=f)
            # np.round(state[8] - state[7], decimals=4)

# first_ep.files