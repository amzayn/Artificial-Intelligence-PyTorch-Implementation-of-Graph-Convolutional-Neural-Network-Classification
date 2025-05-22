import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import random
import numpy as np

# Set seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load dataset with normalization
dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# Train/val/test split
def create_masks(data, val_ratio=0.1, test_ratio=0.2):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    test_size = int(num_nodes * test_ratio)
    val_size = int(num_nodes * val_ratio)
    train_size = num_nodes - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data

data = create_masks(data)

# GCN model with enhancements
class GCN(nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.conv3 = GCNConv(hidden2, out_channels)

        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.act1 = nn.ELU()
        self.act2 = nn.ELU()

        self.reset_parameters()

    def reset_parameters(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.dropout1(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x2 = self.dropout2(x2)

        x3 = self.conv3(x2, edge_index)

        if self.skip is not None:
            x3 = x3 + self.skip(x)

        return F.log_softmax(x3, dim=1)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_features, 256, 128, dataset.num_classes).to(device)
data = data.to(device)

# Optimizer
optimizer = Adam([
    {'params': model.conv1.parameters(), 'weight_decay': 5e-4},
    {'params': model.conv2.parameters(), 'weight_decay': 5e-4},
    {'params': model.conv3.parameters(), 'weight_decay': 5e-4},
    {'params': model.bn1.parameters(), 'weight_decay': 0},
    {'params': model.bn2.parameters(), 'weight_decay': 0},
    {'params': model.skip.parameters(), 'weight_decay': 5e-4} if model.skip is not None else [],
], lr=0.01)

# Scheduler without verbose (for older PyTorch versions)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

# Training step
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    return loss.item()

# Evaluation
@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = pred.eq(data.y[mask]).sum().item()
    acc = correct / mask.sum().item()

    # Class-wise accuracy
    classes = torch.unique(data.y[mask])
    class_acc = {}
    for c in classes:
        class_mask = mask & (data.y == c)
        if class_mask.sum() > 0:
            class_correct = pred[class_mask[mask]].eq(data.y[class_mask]).sum().item()
            class_acc[int(c)] = class_correct / class_mask.sum().item()

    return acc, class_acc

# Training loop
best_val_acc = 0
patience = 100
counter = 0
best_model = None
best_test_acc = 0

for epoch in range(1, 1001):
    loss = train()
    val_acc, val_class_acc = evaluate(data.val_mask)
    test_acc, test_class_acc = evaluate(data.test_mask)

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_model = model.state_dict()
        counter = 0

        print("\nClass-wise Validation Accuracy:")
        for cls, acc in sorted(val_class_acc.items()):
            print(f"Class {cls}: {acc:.4f}")
    else:
        counter += 1

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
              f"Val Acc: {val_acc:.4f} (Best: {best_val_acc:.4f}), "
              f"Test Acc: {test_acc:.4f} (Best: {best_test_acc:.4f})")

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}!")
        break

# Load best model and evaluate
if best_model is not None:
    model.load_state_dict(best_model)
final_test_acc, final_class_acc = evaluate(data.test_mask)
print(f"\n🎯 Final Test Accuracy: {final_test_acc:.4f}")
print("Class-wise Test Accuracy:")
for cls, acc in sorted(final_class_acc.items()):
    print(f"Class {cls}: {acc:.4f}")

#zain
