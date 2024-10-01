import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas_dataset import PandasDataset
import h5py

torch.manual_seed(42)

class SimpleNN(nn.Module):
    def __init__(self, num_unique_customer, embedding_dim, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(num_unique_customer, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + input_size, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, output_size)  # Output size 2 for binary classification
        #self._initialize_weights()

    def forward(self, x_customer, x_other):
        embedded_customer = self.embedding(x_customer)
        x = torch.cat((embedded_customer, x_other), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.5)

        # Initialize biases to small random values
        nn.init.constant_(self.fc3.bias, 0.5)  # Encouraging balanced predictions


# Define a custom weight initialization function
def custom_weight_init(m):
    if isinstance(m, nn.Linear):
        # Example: Initialize with Xavier Uniform
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)  # Initialize bias as zeros



df = pd.read_csv('data/processed_order_store.csv')
# Drop rows where any column has None, NaN, or an empty string
df = df.dropna()
df = df[(df != '').all(axis=1)]
df = df.sample(frac=0.4)
#print(df.head())

# max date feature: 15340
df['date'] = df['date'] / 15340

# Separate features and label
y = df['trip_type'].values
X_embed = df['o_customer_sk'].values
X = df.drop(["o_order_id", "o_customer_sk", "trip_type"], axis=1)
X = X.values

# Split data into train and test sets
#X_embed_train, X_embed_test, X_train, X_test, y_train, y_test = train_test_split(X_embed, X, y, test_size=0.5, random_state=42)

# Convert data to PyTorch tensors
#X_embed_train_tensor = torch.tensor(X_embed_train, dtype=torch.long)  # For embedding
#X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#X_embed_test_tensor = torch.tensor(X_embed_test, dtype=torch.long)  # For embedding
#X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#y_test_tensor = torch.tensor(y_test, dtype=torch.long)

X_embed_tensor = torch.tensor(X_embed, dtype=torch.long)  # For embedding
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create DataLoader for batching
batch_size = 32  # or use 8
train_data = TensorDataset(X_embed_tensor, X_tensor, y_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#test_data = TensorDataset(X_test_tensor, y_test_tensor)
#test_loader = DataLoader(test_data, batch_size=batch_size)

model = SimpleNN(70710, 16, 77, 1000)
#model.apply(custom_weight_init)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Training loop with validation
epochs = 6
lowest_loss = None
for epoch in range(epochs):
    model.train()
    
    # Train in batches
    epoch_loss = 0
    num_batches = 0
    correct = 0
    total = 0
    for X_embed_batch, X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_embed_batch, X_batch)
        loss = criterion(outputs, y_batch)
        epoch_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        num_batches += 1
        _, predicted = outputs.max(1)  # Convert probabilities to binary predictions
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    train_loss = epoch_loss/num_batches
    accuracy = (correct / total) * 100

    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.4f}%')
    if lowest_loss is None or train_loss < lowest_loss:
        lowest_loss = train_loss
        # Save model weights
        torch.save(model.state_dict(), 'models/trip_type_classify.pth')
        print("Best model saved")


model = SimpleNN(70710, 16, 77, 1000)
model.load_state_dict(torch.load('models/trip_type_classify.pth'))
model.eval()
with h5py.File('models/trip_type_classify.h5', 'w') as h5f:
    # Loop through each layer and save the weights and biases
    for name, param in model.named_parameters():
        wb = param.detach().cpu().numpy()
        
        if name == 'fc1.weight':
            w11 = wb[:, :24].T
            w12 = wb[:, 24:].T
            h5f.create_dataset("w11", data=w11)
            h5f.create_dataset("w12", data=w12)

        if "weight" in name and "embedding" not in name:
            wb = wb.T
        h5f.create_dataset(name, data=wb)
        print(name, wb.shape)








