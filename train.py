import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(dataset, model, optimizer, criterion):
    train_target, train_pred, train_loss = [],[],[]
    model.train()

    for x_batch, y_batch in tqdm(dataset):
        optimizer.zero_grad()               # Zero the gradients        
        outputs = model(x_batch)            # Forward Pass - compute the predicted o/p by passing inputs to model        
        loss = criterion(outputs, y_batch)  # loss computation

        loss.backward()
        optimizer.step()

        train_target.append(y_batch.cpu().detach().numpy())
        train_pred.append(outputs.cpu().detach().numpy())
        train_loss.append(loss.cpu().detach().numpy())

    train_target = np.concatenate(train_target, axis=0)
    train_pred = np.concatenate(train_pred, axis=0)
    train_attr = {
        'target' : train_target,
        'predictions' : train_pred,
        'loss' : train_loss
    }
    return train_attr

def eval_model(dataset, model, optimizer, criterion):
    val_target, val_pred, val_loss = [],[],[]
    model.eval()

    for x_batch, y_batch in tqdm(dataset):
        # optimizer.zero_grad()               # Zero the gradients        
        outputs = model(x_batch)            # Forward Pass - compute the predicted o/p by passing inputs to model        
        loss = criterion(outputs, y_batch)  # loss computation

        val_target.append(y_batch.cpu().detach().numpy())
        val_pred.append(outputs.cpu().detach().numpy())
        val_loss.append(loss.cpu().detach().numpy())

    val_target = np.concatenate(val_target, axis=0)
    val_pred = np.concatenate(val_pred, axis=0)
    val_attr = {
        'target' : val_target,
        'predictions' : val_pred,
        'loss' : val_loss
    }
    return val_attr

def save_model(path, epoch, model, optimizer, train_loss=None, train_mse=None, val_loss=None, val_mse=None):
    model_save_path = os.path.join(path, f'model_epoch_{epoch}.pth')
    # Checkpointing: save model, optimizer, and metrics
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_mse': train_mse,
        'val_loss': val_loss,
        'val_mse': val_mse,
    }
    torch.save(checkpoint, model_save_path)

def save_best_model(path, epoch, model, optimizer, train_loss=None, train_mse=None, val_loss=None, val_mse=None):
    best_model = os.path.join(path, 'best_model.pth')

    if os.path.exists(best_model) is False:
        best_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_mse': train_mse,
            'val_loss': val_loss,
            'val_mse': val_mse,
        }
        torch.save(best_checkpoint, best_model)
    else:
        checkpoint = torch.load(best_model)
        best_val_loss = checkpoint['val_loss']

        if val_loss < best_val_loss:            
            os.remove(best_model)       # Current model is better, update the best model
        else:            
            return                      # Existing model is better or equal, do not update
        
        best_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_mse': train_mse,
            'val_loss': val_loss,
            'val_mse': val_mse,
        }
        torch.save(best_checkpoint, best_model)
        


def load_best_model(path, model, optimizer):
    files = os.listdir(path)
    if files:
        best_model = os.path.abspath(os.path.join(path, files[0]))
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint.get('train_loss', None)
    train_mse = checkpoint.get('train_mse', None)
    val_loss = checkpoint.get('val_loss', None)
    val_mse = checkpoint.get('val_mse', None)
    return epoch, train_loss, train_mse, val_loss, val_mse

class Model_training(object):
    def __init__(self, trainset, valset, run_dir, output_loc, model, use_pretrained_model, total_epochs): 
        self.run_dir = run_dir
        self.epochs = total_epochs
        self.output_loc = output_loc
        start_epoch = 1
        use_pt_model = use_pretrained_model


        # Load the Model
        model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)       
        criterion = nn.MSELoss()

        # Output Locations
        # 1. Metrics
        metrics_path = os.path.join(self.run_dir, 'Metrics')
        if os.path.exists(metrics_path) is False:
            os.mkdir(metrics_path)

        # 2. Checkpoint Location
        checkpoint_path = os.path.join(self.output_loc, 'Checkpoints')
        if os.path.exists(checkpoint_path) is False:
            os.mkdir(checkpoint_path)

        # Loss and MSE variables
        train_loss, train_mse = [],[]
        val_loss, val_mse = [],[]
        best_val_loss = np.inf
        best_train_loss = np.inf

        # Use Pre-trained model if the check is enabled in main.py
        if use_pt_model is not False:
            best_model_path = os.path.join(checkpoint_path, "best_train")
            epoch, train_loss, train_mse, old_val_loss, old_val_mse = load_best_model(best_model_path, model, optimizer)

            # Run validation once to cross verify the validation metrics
            val_attr = eval_model(valset, model, optimizer)
            avg_val_loss = np.average(val_attr['loss'])
            mse_val = np.mean((val_attr['target'] - val_attr['predictions']) ** 2)
            val_loss.append(np.average(val_attr['loss']))
            val_mse.append(np.mean((val_attr['target'] - val_attr['predictions']) ** 2))
            print(f"Train Loss: {avg_val_loss} \n Train Mean Sqaure Error (MSE) : {mse_val}")
            print('Validation Complete')
        
        # Train the dataset and run the trained model with validation data
        else:
            # Running epochs for training the model        
            for epoch in range(start_epoch, self.epochs):
                print(f"----Training for Epoch:{epoch} ")

                # Train dataset
                train_attr = train_model(trainset, model, optimizer, criterion)
                avg_train_loss = np.average(train_attr['loss'])
                mse_train = np.mean((train_attr['target'] - train_attr['predictions']) ** 2)
                train_loss.append(np.average(train_attr['loss']))
                train_mse.append(np.mean((train_attr['target'] - train_attr['predictions']) ** 2))
                print(f"Train Loss: {avg_train_loss} ||| Train Mean Sqaure Error (MSE) : {mse_train}\n")

                # Validation dataset
                val_attr = eval_model(valset, model, optimizer, criterion)
                avg_val_loss = np.average(val_attr['loss'])
                mse_val = np.mean((val_attr['target'] - val_attr['predictions']) ** 2)
                val_loss.append(np.average(val_attr['loss']))
                val_mse.append(np.mean((val_attr['target'] - val_attr['predictions']) ** 2))
                print(f"Validation Loss: {avg_val_loss} ||| Validation Mean Sqaure Error (MSE) : {mse_val}")
                

                # Save the metrics for train and validation for every epoch
                
                with open(os.path.join(metrics_path, "metric_%d.txt" % epoch), 'w') as ff:
                    ff.write(f"**** Training Metrics **** \nTrain Loss = {avg_train_loss}\nTrain MSE = {mse_train}\
                            \n\n**** Validation Metrics ****\nValidation Loss = {avg_val_loss}\nValidation MSE = {mse_val}")

                # Save the models              
                if np.mean(val_attr["loss"]) < best_val_loss:
                    best_val_loss = np.mean(val_attr["loss"])
                    save_model(checkpoint_path, epoch, model, optimizer, train_loss=train_loss[-1], train_mse=train_mse[-1],
                                val_loss=val_loss[-1], val_mse=val_mse[-1])
                # Best Model
                elif np.mean(train_attr["loss"]) < best_train_loss:
                    best_train_loss = np.mean(train_attr["loss"])
                    save_best_model(os.path.join(checkpoint_path, "best_train"), epoch,model, optimizer, train_loss=train_loss[-1],
                                train_mse=train_mse[-1], val_loss=val_loss[-1], val_mse=val_mse[-1])
        
            print(f"Mean Train Loss = {np.mean(train_loss)}")
            print(f"Mean Validation Loss = {np.mean(val_loss)}")


        # Save and plot Epoch-Loss        
        # train and validation data
        train_loss = np.array(train_loss)
        train_mse = np.array(train_mse)
        val_loss = np.array(val_loss)
        val_mse = np.array(val_mse)
        loss_all = np.concatenate((train_loss[:, np.newaxis], val_loss[:, np.newaxis]), axis=1)
        mse_all = np.concatenate((train_mse[:, np.newaxis], val_mse[:, np.newaxis]), axis=1)

        # Epoch - Loss Graph
        fig_loss = plt.figure(num="loss", dpi=90, figsize=(16, 9))
        plt.plot(range(train_loss.shape[0]), train_loss, "-b", linewidth=0.5, label="train_loss")
        plt.plot(range(val_loss.shape[0]), val_loss, "-r", linewidth=0.5, label="val_loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        fig_loss.savefig(os.path.join(self.run_dir, "epoch_loss.png"))

        # Epoch - MSE Graph
        fig_mse = plt.figure(num='mse', dpi=90, figsize=(16, 9))
        plt.plot(range(train_mse.shape[0]), train_mse, "-b", linewidth=0.5, label="train_mse")
        plt.plot(range(val_mse.shape[0]), val_mse, "-r", linewidth=0.5, label="val_mse")
        plt.ylabel("Mean Sqaured Error")
        plt.xlabel("Epoch")
        plt.legend()
        fig_mse.savefig(os.path.join(self.run_dir, "epoch_mse.png"))

        # Save metrics for MSE and Loss
        np.savetxt(os.path.join(self.run_dir, "epoch_loss.txt"), loss_all, delimiter=',', )
        np.savetxt(os.path.join(self.run_dir, "epoch_mse.txt"), mse_all, delimiter=',', )

        print("Training complete.")
