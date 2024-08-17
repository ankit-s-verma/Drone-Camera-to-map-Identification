import dataset as dt
import LSTM_Network_Model
import train
import test
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch, torch.nn as nn

trainset_location = './data_train'              # Train dataset location
validationset_location = './data_val'           # Validation dataset location
testset_location = './data_test'                # Test dataset location
output_location = './output'                    # Output folder location
train_model = True
test_model = True
pretrain_model = False
total_epoch = 31

def get_folder_list(path):
    folders = []    
    names = os.listdir(path)
    for name in names:
        data_path = os.path.join(os.path.abspath(path), name)
        if os.path.isdir(data_path):    
            folders.append(data_path)
    folders.sort()
    return folders

if os.path.exists(output_location) is False:
    os.mkdir(output_location)

if __name__ == "__main__":

    # Initializing the run
    run_name = input('Run Name:')
    run_dir = os.path.join(output_location, 'Runs', run_name)
    if os.path.exists(run_dir) is False:
        os.mkdir(run_dir)

    # Loading Train set
    train_data_path = get_folder_list(trainset_location)
    train_basic = dt.DataInitialization(train_data_path, mode='train')
    train_dataset = dt.SequenceDataset(train_basic, train_basic.get_merged_index_map())
    trainset = DataLoader(train_dataset, batch_size=512, shuffle=False)

    # Loading Validation set
    val_data_path = get_folder_list(validationset_location)
    val_basic = dt.DataInitialization(val_data_path, mode='val')
    validation_dataset = dt.SequenceDataset(val_basic, val_basic.get_merged_index_map())
    valset = DataLoader(validation_dataset, batch_size=512, shuffle=False)

    # Build the model
    model = LSTM_Network_Model.load_model()

    # Train the model
    if train_model:
        train.Model_training(trainset, valset, run_dir, output_location, model, pretrain_model, total_epoch)
    
    if test_model:
        # Load the best model
        output_location = './output'
        checkpoints_folder = os.path.join(output_location, "Checkpoints")
        best_model_filename = 'best_model.pth'
        best_model_path = os.path.join(checkpoints_folder, best_model_filename)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Check if we have the best model, else select the last checkpoint created
        if os.path.exists(best_model_path):
            load_model = torch.load(best_model_path)
            print(f"Loading Best Model - {best_model_filename}")
        else:
            # Find the last created checkpoint
            checkpoint_files = os.listdir(checkpoints_folder)
            last_checkpoint = sorted(checkpoint_files)[-1]
            load_model = torch.load(os.path.join(checkpoints_folder, last_checkpoint))
            print(f"Loading last saved model - {last_checkpoint}")

        model.load_state_dict(load_model.get('model_state_dict'))
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)       
        criterion = nn.MSELoss()
        test.tester(model, testset_location, run_dir, optimizer, criterion)

