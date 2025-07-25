from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import h5py


# ====> DEBUG
# import debugpy
# debugpy.listen(5678)
# print('waiting for client to attach')
# debugpy.wait_for_client()  # blocks execution until client is attached


class YbqFloorSmall(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ybq floor small."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")        

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, demo_id):
            # load raw data
            with h5py.File(episode_path, "r") as F:
                if f"demo_{demo_id}" not in F['data'].keys():
                    return None # skip episode if the demo doesn't exist (e.g. due to failed demo)
                actions = F['data'][f"demo_{demo_id}"]["actions"][()]
                states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
                gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
                joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
                images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
                wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

            # compute language instruction
            raw_file_string = os.path.basename(episode_path).split('/')[-1]
            words = raw_file_string[:-10].split("_")
            command = ''
            for w in words:
                if "SCENE" in w:
                    command = ''
                    continue
                command = command + w + ' '
            command = command[:-1]

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(actions.shape[0]):
                # # => DEBUG front image: 
                # from PIL import Image
                # im = Image.fromarray(images[i][::-1,::,::])
                # im.save('test_obs.png')
                # # => DEBUG wrist image:
                # from PIL import Image
                # im = Image.fromarray(wrist_images[i][::-1,::,::])
                # im.save('test_wrist.png')

                episode.append({
                    'observation': {
                        'image': images[i][::-1,::,::],
                        'wrist_image': wrist_images[i][::-1,::,::],
                        'state': np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                        'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                    },
                    'action': np.asarray(actions[i], dtype=np.float32),
                    'discount': 1.0,
                    'reward': float(i == (actions.shape[0] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (actions.shape[0] - 1),
                    'is_terminal': i == (actions.shape[0] - 1),
                    'language_instruction': command,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path + f"_{demo_id}", sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing

        for sample in episode_paths:
            with h5py.File(sample, "r") as F:
                n_demos = len(F['data'])
            idx = 1
            cnt = 1
            while cnt < (n_demos+1):
                ret = _parse_example(sample, idx)
                if ret is not None:
                    cnt += 1
                idx += 1
                yield ret

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

