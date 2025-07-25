from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os


class RobosuiteLiftRedBlock(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for robosuite lift red block dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        
        import debugpy
        debugpy.listen(5678)
        print('waiting for client to attach')
        debugpy.wait_for_client()  # blocks execution until client is attached


    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(10,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
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
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
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
            'train': self._generate_examples(path='data/train/ep_*'),
            'val': self._generate_examples(path='data/val/ep_*'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset

            episode_frags = sorted(glob.glob(os.path.join(episode_path,'state_*_*.npz')))

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []

            total_lenght = 


            for frag, episode_frag in enumerate(episode_frags):

                frag_data = np.load(episode_frag, allow_pickle=True)     # this is a list of dicts in our case

                for i, step in enumerate(frag_data.items()):
                    # compute Kona language embedding

                    #TODO: language instruction direttamente nel file .npz
                    language_instruction = 'lift the red block'
                    language_embedding = self._embed([language_instruction])[0].numpy()

                    last = (frag+1)*i == (len(episode_frags))*(len(frag_data['states']) - 1)
                    episode.append({
                        'observation': {
                            'image': frag_data['image_obs'][i],
                            # 'wrist_image': step['wrist_image'],
                            'state': np.concatenate([frag_data['gripper_state'][i], np.array([frag_data['action_infos'][i]['actions'][-1]])])
                        },
                        'action': np.concatenate([frag_data['gripper_state'][i+1], np.array([frag_data['action_infos'][i+1]['actions'][-1]])]),
                        'discount': 1.0,
                        'reward': float(i == (len(frag_data) - 1)),
                        'is_first': i == 0,
                        'is_last': i == (len(frag_data) - 1),
                        'is_terminal': i == (len(frag_data) - 1),
                        'language_instruction': language_instruction,
                        'language_embedding': language_embedding,
                    })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

