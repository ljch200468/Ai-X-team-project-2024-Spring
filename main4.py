import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.dirname(__file__))
print("CWD:", os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataset import CustomDataset  # Assuming you have defined your dataset class in custom_dataset.py
from model import TwoBranchCNNLSTM  # Assuming you have defined your model in model.py
from raft_module import get_raft_module, estimate_flow
from tqdm import tqdm
import csv
from sklearn.metrics import confusion_matrix

print(torch.cuda.is_available())


import warnings
warnings.filterwarnings("ignore")

from dataset3 import (
    CustomDataset,
)  # Assuming you have defined your dataset class in custom_dataset.py
from model import TwoBranchCNNLSTM  # Assuming you have defined your model in model.py
from raft_module import get_raft_module, estimate_flow

def train(model, flownet, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    predicted_labels = []
    true_labels = []
    pbar = tqdm(total= len(train_loader))
    for batch_idx, (sequence_in, sequence_out, target, category) in enumerate(train_loader):

        lof_in, lof_out = [], []
        with torch.no_grad():
            for i in range(len(sequence_in) - 1):
                lof_in.append(estimate_flow(flownet, torch.tensor(sequence_in[i]).to(device), torch.tensor(sequence_in[i + 1]).to(device)))
                lof_out.append(estimate_flow(flownet, torch.tensor(sequence_out[i]).to(device), torch.tensor(sequence_out[i + 1]).to(device)))
            target = target.to(device)

        optimizer.zero_grad()

        # Forward pass through each time step
        state = None
        for t in range(len(lof_in)):
            output, state = model(lof_in[t], lof_out[t], state)

        # Calculate loss
        loss = criterion(output, category.to(device))
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = output.max(1)
        tmp = output.argmax(1) == category.to(device).argmax(1)
        tmp.to(torch.int).sum().item()
        total_correct += tmp.to(torch.int).sum().item()

        predicted_labels.extend(predicted.tolist())
        true_labels.extend(category.argmax(dim=1).tolist())  # Convert one-hot encoded labels to class labels

        pbar.update(1)

    pbar.close()

    return total_loss / len(train_loader.dataset), total_correct / len(train_loader.dataset), predicted_labels, true_labels
    #return total_loss / len(train_loader.dataset), total_correct / len(train_loader.dataset)

def evaluate(model, flownet, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch_idx, (sequence_in, sequence_out, target, category) in enumerate(val_loader):

            lof_in, lof_out = [], []
            for i in range(len(sequence_in) - 1):
                lof_in.append(estimate_flow(flownet, torch.tensor(sequence_in[i]).to(device), torch.tensor(sequence_in[i + 1]).to(device)))
                lof_out.append(estimate_flow(flownet, torch.tensor(sequence_out[i]).to(device), torch.tensor(sequence_out[i + 1]).to(device)))
            target = target.to(device)

            # Forward pass through each time step
            state = None
            for t in range(len(lof_in)):
                output, state = model(lof_in[t], lof_out[t], state)

            # Calculate loss
            loss = criterion(output, category.to(device))
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = output.max(1)
            tmp = output.argmax(1) == category.to(device).argmax(1)
            tmp.to(torch.int).sum().item()
            total_correct += tmp.to(torch.int).sum().item()

            predicted_labels.extend(predicted.tolist())
            true_labels.extend(category.argmax(dim=1).tolist())     # Convert one-hot encoded labels to class labels

    return total_loss / len(val_loader.dataset), total_correct / len(val_loader.dataset), predicted_labels, true_labels
    #return total_loss / len(val_loader.dataset), total_correct / len(val_loader.dataset)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.0001
    num_epochs = 15
    hidden_size = 64
    num_categories = 5  # Number of categories in the dataset
    T = 14  # Sequence length

    # Directories
    train_data_dir = "./SampleData/"
    val_data_dir = "./SampleDataTest/"
    ckpt_dir = "./checkpoints/"
    os.makedirs(ckpt_dir, exist_ok= True)

    # Initialize model
    model = TwoBranchCNNLSTM(input_channels=2, hidden_size=hidden_size, num_ops=num_categories).to(device)
    flownet = get_raft_module().cuda()

    # Initialize dataset and data loaders
    train_dataset = CustomDataset(root_dir=train_data_dir, T=T)
    val_dataset = CustomDataset(root_dir=val_data_dir, T=T)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_predicted_labels, train_true_labels = train(model, flownet, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_predicted_labels, val_true_labels = evaluate(model, flownet, val_loader, criterion, device)

        #train_loss, train_accuracy = train(model, flownet, train_loader, criterion, optimizer, device)
        #val_loss, val_accuracy = evaluate(model, flownet, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Calculate confusion matrix
        train_conf_matrix = confusion_matrix(train_true_labels, train_predicted_labels)
        val_conf_matrix = confusion_matrix(val_true_labels, val_predicted_labels)

        print("Train Confusion Matrix:")
        print(train_conf_matrix)
        print("Validation Confusion Matrix:")
        print(val_conf_matrix)

        # Save the trained model
        torch.save(model.state_dict(), ckpt_dir + 'trained_model-' + str(epoch) + '.pth')

    # Save losses to CSV file
    #with open('loss_data.csv', mode='w', newline='') as file:
        #writer = csv.writer(file)
        #writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        #for epoch in range(num_epochs):
            #writer.writerow([epoch+1, train_loss[epoch], val_loss[epoch]])



if __name__ == '__main__':
    main()
