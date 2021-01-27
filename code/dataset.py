import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader


class NTUDataset(Dataset):
    
    def __init__(self, sample_set, params, transform=None):
        # Initialize all parameters for the model
        self.sample_set = sample_set
        self.kp_shape = params.kp_shape
        self.seg_size = params.seg_size
        self.data_path = params.data_path
        self.num_channels = params.num_channels
        self.transform = transform
        self.mean, self.std = self.get_mean_std()
    
    def __len__(self):
        # Number of samples in the dataset
        return len(self.sample_set)
    
    def __getitem__(self, idx):
        # Return a particular item from the dataset
        sample_name = self.sample_set[idx]
        
        # Process the sample into tensor keypoints for the given index
        sample_kp, action_class = self.read_sample(sample_name)
        sample_kp = torch.tensor(sample_kp)
        
        if self.transform:
            sample_kp = self.transform(sample_kp)
            
        # Normalize with mean and std
        sample_kp -= self.mean
        sample_kp /= self.std
        
        return sample_kp, action_class
    
    # ----- Helper functions -----
    def read_sample(self, sample_name):
        sample_path = os.path.join(self.data_path, sample_name)
        data = np.load(sample_path, allow_pickle=True).item()
        # Each data sample has the following keys:
        # dict_keys(['file_name', 'nbodys', 'njoints', 'skel_body0', 'rgb_body0', 'depth_body0', 'skel_body1', 'rgb_body1', 'depth_body1'])
        # For now, I am just considering one participant for each video segment and taking 'skel_body0' as input keypoints
        kps = self.augment_kp(data['skel_body0'])
        # Subtracting -1 from the action_class to make it compatible with Pytorch labels (starts from 0 to 119)
        action_class = int(sample_name.split('A')[1][:3]) - 1
        # Before returning the sample_kp, change its shape in the form: (seg_size, 1, 25, 3)
        # This is to treat each frame as a form of single channel input image
        return np.reshape(kps, (self.seg_size, 1, self.kp_shape[0], self.kp_shape[1])), action_class
    
    def augment_kp(self, sample_kp):
        # Temporally augment video segment based on the minimum segment size for the dataset
        # Randomly take "seg_size" number of frames from the segment (in chronological order)
        sample_size = sample_kp.shape[0]
        if sample_size < self.seg_size:
            # Pad same frames at the end in order to meet the segment size requirement
            return self.pad_frames(sample_kp)
        rand_segments = sorted(random.sample(range(0, sample_size), self.seg_size))
        sample_kp = np.take(sample_kp, rand_segments, axis=0)
        return sample_kp
    
    def pad_frames(self, sample_kp):
        # Consider seg_size for the dataset is 40 and the current sample has only 15 frames in the segment
        # We will need to repeat the frames in order to make it reach 40
        padded_kp = sample_kp
        sample_size = sample_kp.shape[0]
        additional_frames = self.seg_size - sample_size
        while additional_frames >= sample_size:
            padded_kp = np.concatenate((padded_kp, sample_kp), axis=0)
            additional_frames -= sample_size
            
        padded_kp = np.concatenate((padded_kp, sample_kp[:additional_frames]), axis=0)
        return padded_kp
    
    def get_mean_std(self):
        mean_path = f'../mean_std/mean_{self.seg_size}_{self.kp_shape[0]}_{self.kp_shape[1]}.npy'
        std_path = f'../mean_std/std_{self.seg_size}_{self.kp_shape[0]}_{self.kp_shape[1]}.npy'
        try:
            # Read pickled file for mean and std
            mean = torch.from_numpy(np.load(mean_path))
            std = torch.from_numpy(np.load(std_path))
        except OSError:
            print('Evaluating mean and std for the training set...')         
            X = torch.tensor([self.read_sample(sample_name)[0] for sample_name in self.sample_set])
            X = X.view(-1, self.seg_size, self.num_channels, self.kp_shape[0], self.kp_shape[1])
            mean = torch.mean(X, axis=0)
            std = torch.std(X, axis=0)
            np.save(mean_path, mean.numpy())
            np.save(std_path, std.numpy())
            print(f'Mean and Std saved successfully!')
            
        return mean, std
            



# samples file_name = 'S018C001P042R002A120.skeleton.npy'
# P042 is the participant number
# Remember that I am trying to split the dataset based on the participants and not the total samples
# This means that the validation set will have samples from all unique participants that are not involved in the train set

def get_participant_number(file_name):
    return file_name.split('P')[1][:3]

def split_participants(data_path, val_pct=0.2):
    # Returns a random list of participants for the train and validation sets respectively
    samples = os.listdir(data_path)
    total_samples = len(samples)
    # Get all unique participant numbers
    all_participants = set()
    for sample in samples:
        part = get_participant_number(sample)
        all_participants.add(part)
    total_participants = len(all_participants)
    all_participants = list(all_participants)
    
    # Split into train and val sets
    val_len = int(total_participants * val_pct)
    # Randomly shuffle the list
    random.shuffle(list(all_participants))
    train_participants = all_participants[val_len:]
    val_participants = all_participants[:val_len]

    print(f'Total Video Samples: {len(samples)} || Total Participants: {len(all_participants)} || Train Participants: {len(train_participants)} || Validation Participants: {len(val_participants)}')
    return train_participants, val_participants

def get_train_val_set(data_path, val_pct=0.2, temporal_aug_k=3):
    train_participants, val_participants = split_participants(data_path, val_pct)
    train_samples, val_samples = [], []
    # min_seg_size = 1000
    for sample in os.listdir(data_path):
        participant_number = get_participant_number(sample)
        # Temporary code to check the minimum segment size in the dataset
        # data = np.load(os.path.join(data_path, sample), allow_pickle=True).item()['skel_body0']
        # min_seg_size = min(min_seg_size, data.shape[0])
        
        # Apply data augmentation here ('k' times random temporal augmentation)
        for _ in range(temporal_aug_k):
            if participant_number in val_participants:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
    
    # print(f'Minimum segment size in the dataset: {min_seg_size}')
    return train_samples, val_samples



if __name__ == '__main__':
    # Sample code on how to load the dataset and the loader
    train_samples, val_samples = get_train_val_set(data_path=data_dir, val_pct=0.2, temporal_aug_k=3)
    print(f'Train samples: {len(train_samples)} || Validation samples: {len(val_samples)}')
    
    # Load train and validation dataset
    train_set = NTUDataset(data_path=data_dir, sample_set=train_samples)
    val_set = NTUDataset(data_path=data_dir, sample_set=val_samples)

    BATCH_SIZE = 8
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)