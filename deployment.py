import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                           n_redundant=2, weights=[0.8, 0.2], random_state=42)

def balance_dataset(X, y):
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    y_majority = y[y == 0]
    y_minority = y[y == 1]

    # Oversample clase minoritaria
    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=len(y_majority),
        random_state=42
    )

    # Combinar
    X_balanced = np.vstack((X_majority, X_minority_upsampled))
    y_balanced = np.hstack((y_majority, y_minority_upsampled))

    return X_balanced, y_balanced

X_bal, y_bal = balance_dataset(X, y)

# Escalado
X_bal = StandardScaler().fit_transform(X_bal)
y_bal = y_bal.reshape(-1, 1)

X_tensor = torch.tensor(X_bal, dtype=torch.float32)
y_tensor = torch.tensor(y_bal, dtype=torch.float32)


class ShallowNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super(ShallowNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.out_act(self.fc2(x))

bce_loss = nn.BCELoss()

def weighted_bce(preds, target, weight_pos=2.0):
    weights = torch.where(target == 1, weight_pos, 1.0)
    bce = nn.BCELoss(reduction='none')
    loss = bce(preds, target)
    return torch.mean(loss * weights)

def train(model, loss_fn, X, y, lr=0.01, epochs=500):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

model_bce = ShallowNNClassifier(input_dim=X.shape[1])
train(model_bce, bce_loss, X_tensor, y_tensor)

model_asym = ShallowNNClassifier(input_dim=X.shape[1])
train(model_asym, lambda pred, y: weighted_bce(pred, y, weight_pos=3.0), X_tensor, y_tensor)

# Interpretación con método Olden 
def olden_importance(model):
    with torch.no_grad():
        W1 = model.fc1.weight.detach().numpy()
        W2 = model.fc2.weight.detach().numpy()
        importance = np.dot(W2, W1)
    return importance.flatten()

importance_bce = olden_importance(model_bce)
importance_asym = olden_importance(model_asym)


features = [f"X{i}" for i in range(X.shape[1])]
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(features, importance_bce)
plt.title("Importancia - BCE estándar")
plt.ylabel("Importancia")

plt.subplot(1, 2, 2)
plt.bar(features, importance_asym)
plt.title("Importancia - BCE Asimétrica")

plt.tight_layout()
plt.show()
