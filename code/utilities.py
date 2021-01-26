import argparse
from easydict import EasyDict as edict


"""
- kp_shape = (25,3)
- seg_size = varies based on the action being performed (so, select a minimum segment size among all samples in the dataset)
- participant_list <= those who are in the train or validation or test set (a list of numbers/codes for the participants)
- data_path = '/data/zak/graph/ntu/train'
- BATCH_SIZE <== For the model
- temporal_aug_k <== Defines number of random samples from one segment (for temporal augmentation)
"""

parser = argparse.ArgumentParser(description='NTU Activity Recognition with 3D Keypoints')
parser.add_argument('--data_path', type=str, default='/data/zak/graph/ntu/train', help='Dataset path')
parser.add_argument('--seg_size', type=int, default=50, help='Minimum segment size for each video segment')
parser.add_argument('--kp_shape', type=int, nargs=2, default=[25, 3], help='(n_joints, n_coordinates) -- (25, 3)')
parser.add_argument('--BATCH_SIZE', type=int, default=8, help='Batch size for the dataset')
parser.add_argument('--temporal_aug_k', type=int, default=3, help='Number of temporal augmentations for each sample')


parsed_input = parser.parse_args()

params = edict({
    'kp_shape': parsed_input.kp_shape,
    'seg_size': parsed_input.seg_size,
    'data_path': parsed_input.data_path,
    'BATCH_SIZE': parsed_input.BATCH_SIZE,
    'temporal_aug_k': parsed_input.temporal_aug_k,
})

print(params)