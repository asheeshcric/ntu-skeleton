import argparse
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from dataset import NTUDataset, get_train_val_set
from model import ConvLSTM


def get_train_val_loader(params, val_pct=0.2):
    train_samples, val_samples = get_train_val_set(data_path=params.data_path, val_pct=val_pct, temporal_aug_k=params.temporal_aug_k)
    print(f'Train samples: {len(train_samples)} || Validation samples: {len(val_samples)}')
    
    # Apply transform to normalize the data
    # transform = transforms.Normalize((0.5), (0.5))
    
    # Load train and validation dataset
    train_set = NTUDataset(sample_set=train_samples, params=params, transform=None)
    val_set = NTUDataset(sample_set=val_samples, params=params, transform=None)

    train_loader = DataLoader(train_set, batch_size=params.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params.BATCH_SIZE, shuffle=True)
    
    return train_loader, val_loader


def save_model(model):
    current_time = datetime.now()
    current_time = current_time.strftime("%m_%d_%Y_%H_%M")
    torch.save(model.state_dict(), f'ntu_lstm_{current_time}.pth')
    
    
def build_test_stats(preds, actual, acc, params):
    print(f'Model accuracy: {acc}')
    
    # For confusion matrix
    preds = [int(k) for k in preds]
    actual = [int(k) for k in actual]

    cf = confusion_matrix(actual, preds, labels=list(range(params.num_classes)))

def train(model, train_loader, loss_function, optimizer, params):
    print('Training...')
    for epoch in range(params.n_epochs):
        for batch in tqdm(train_loader):
            inputs, labels = batch[0].to(device).float(), batch[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch} | Loss: {loss}')

    return model

def test(model, test_loader):
    print('Testing...')
    correct = 0
    total = 0

    preds = []
    actual = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device).float(), labels.to(device)
            class_outputs = model(inputs)
            _, class_prediction = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (class_prediction == labels).sum().item()
            preds.extend(list(class_prediction.to(dtype=torch.int64)))
            actual.extend(list(labels.to(dtype=torch.int64)))

    acc = 100*correct/total
    return preds, actual, acc


def main(params):
    # Initialize some variables to track progress
    accs = []
    
    # Initialize the model
    model = ConvLSTM(params=params).to(device)
    
    # Use parallel computing if available
    if device.type == 'cuda' and n_gpus > 1:
        model = nn.DataParallel(model, list(range(n_gpus)))
        
    # Loss Function and Optimizer (can use weight=class_weights if it is a disbalanced dataset)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Get train and validation loaders
    train_loader, val_loader = get_train_val_loader(params, val_pct=0.2)
    
    # Train the model
    model = train(model, train_loader, loss_function, optimizer, params)
    save_model(model)
    
    # Validate the model
    preds, actual, acc = test(model, val_loader)
    build_test_stats(preds, actual, acc, params)
    
    


if __name__ == '__main__':
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
    parser.add_argument('--k_fold', type=int, default=1, help='k-Fold validation')
    parser.add_argument('--n_epochs', type=int, default=8, help='Number of Epochs to train the model')


    parsed_input = parser.parse_args()

    params = edict({
        'kp_shape': parsed_input.kp_shape,
        'seg_size': parsed_input.seg_size,
        'data_path': parsed_input.data_path,
        'BATCH_SIZE': parsed_input.BATCH_SIZE,
        'temporal_aug_k': parsed_input.temporal_aug_k,
        'k_fold': parsed_input.k_fold,
        'n_epochs': parsed_input.n_epochs,
        'num_classes': 120,
        'bcc': 32, # base convolution channels
        'num_channels': 1, # channel for each frame
        'num_joints': 25, # joints used in each frame
        'num_coord': 3, # number of coordinates (x, y, z)

    })
    
    # Check for GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {n_gpus}')
    
    # Run the train/val code
    main(params)
    