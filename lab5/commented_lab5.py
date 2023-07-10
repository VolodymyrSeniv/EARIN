import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# Normalize the input images
x_train_full = x_train_full.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the images
x_train_full = np.reshape(x_train_full, (len(x_train_full), -1))
x_test = np.reshape(x_test, (len(x_test), -1))

# Convert labels to one-hot vectors
y_train_full = torch.tensor(y_train_full).long()
y_train_full = nn.functional.one_hot(y_train_full, num_classes=10).float()
y_test = torch.tensor(y_test).long()
y_test = nn.functional.one_hot(y_test, num_classes=10).float()

# Split the dataset into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Create a PyTorch dataset and dataloader
train_dataset = TensorDataset(torch.tensor(x_train).float(), y_train)
val_dataset = TensorDataset(torch.tensor(x_val).float(), y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self, num_hidden_layers, width, activation):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(784, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for i in range(num_hidden_layers)])
        self.output_layer = nn.Linear(width, 10)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


# Trains the provided `model` using the `train_dataloader` and `validation_dataloader` for the specified number of
# `epochs`.
def train_model(model, learning_rate, batch_size, loss_fn, num_epochs, width):
    # Define the optimizer with the given learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get the loss function from the nn module with the given name
    criterion = getattr(nn, loss_fn)()

    # Initialize empty lists to store the training and validation losses and accuracies
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize variables to store the training loss and accuracy
        train_loss, correct, total = 0, 0, 0

        # Loop over the training data loader
        for x, y in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Make a prediction with the model
            y_pred = model(x)

            # Calculate the loss with the given loss function
            loss = criterion(y_pred, y)

            # Backpropagate the loss and update the model parameters
            loss.backward()
            optimizer.step()

            # Add the loss to the running total
            train_loss += loss.item()

            # Calculate the number of correctly classified samples and the total number of samples
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == torch.argmax(y, dim=1)).sum().item()

        # Calculate the average training loss and accuracy for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc = correct / total
        train_accs.append(train_acc)

        # Set the model to evaluation mode
        model.eval()

        # Initialize variables to store the validation loss and accuracy
        val_loss, correct, total = 0, 0, 0

        # Loop over the validation data loader
        with torch.no_grad():
            for x, y in val_loader:
                # Make a prediction with the model
                y_pred = model(x)

                # Calculate the loss with the given loss function
                loss = criterion(y_pred, y)
                val_loss += loss.item()

                # Calculate the number of correctly classified samples and the total number of samples
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == torch.argmax(y, dim=1)).sum().item()

        # Calculate the average validation loss and accuracy for the epoch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_accs.append(val_acc)

        # Print the training and validation losses and accuracies for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')

    # Plot the training and validation losses and accuracies over the epochs
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Return the training and validation losses and accuracies
    return train_losses, val_losses


# Experiment 1: Varying Learning Rate
learning_rates = [0.001, 0.01, 0.1]
for learning_rate in learning_rates:
    print('')
    print(f"Training model with learning rate: {learning_rate}")
    model = Model(num_hidden_layers=1, width=128, activation='ReLU')
    train_model(model, learning_rate, batch_size=64, loss_fn='CrossEntropyLoss', num_epochs=10, width=128)

# Experiment 2: Varying Mini-batch Size
batch_sizes = [16, 64, 256]
for batch_size in batch_sizes:
    print('')
    print(f"Training model with batch size: {batch_size}")
    model = Model(num_hidden_layers=1, width=128, activation='ReLU')
    train_model(model, learning_rate=0.01, batch_size=batch_size, loss_fn='CrossEntropyLoss', num_epochs=10, width=128)

# Experiment 3: Varying Number of Hidden Layers
num_hidden_layers = [0, 1, 2]
for nhl in num_hidden_layers:
    print(f"Training model with {nhl} hidden layers")
    model = Model(num_hidden_layers=nhl, width=128, activation='ReLU')
    train_model(model, learning_rate=0.01, batch_size=64, loss_fn='CrossEntropyLoss', num_epochs=10, width=128)

# Experiment 4: Change width
widths = [32, 64, 128]
for width in widths:
    print(f"Training model with {width} neurons in the hidden layers")
    model = Model(num_hidden_layers=1, width=width, activation='ReLU')
    train_model(model, learning_rate=0.01, batch_size=64, loss_fn='CrossEntropyLoss', num_epochs=10, width=width)

# Experiment 5: Change loss function
loss_fns = ['MSELoss', 'L1Loss', 'CrossEntropyLoss']
for loss_fn in loss_fns:
    print(f"Training model with {loss_fn} loss function")
    model = Model(num_hidden_layers=1, width=128, activation='ReLU')
    train_model(model, learning_rate=0.01, batch_size=64, loss_fn=loss_fn, num_epochs=10, width=128)
