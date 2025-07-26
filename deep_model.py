import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

class EEG_CNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

def run_deep_learning(features_csv="features.csv", run_filter="01", epochs=50, lr=1e-3, batch_size=16):
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df = df[df["run"] == run_filter]
    df = df.dropna(subset=["BDI"])
    df["label"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    features = [col for col in df.columns if col.startswith(('alpha_', 'beta_', 'theta_', 'delta_')) or col in ['faa', 'spectral_entropy']]
    X = df[features].values
    y = df["label"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [B, 1, L]
    y = torch.tensor(y, dtype=torch.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    train_losses_fold1 = []
    val_accuracies_fold1 = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = EEG_CNN(input_dim=X.shape[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for epoch in range(epochs):
            model.train()
            idx = torch.randperm(len(X_train))
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb, yb = X_train[batch_idx], y_train[batch_idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if fold == 0:
                train_losses_fold1.append(epoch_loss / len(X_train))
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_test).numpy()
                val_acc = accuracy_score(y_test.numpy(), (val_preds > 0.5).astype(int))
                val_accuracies_fold1.append(val_acc)

        # Final fold evaluation
        model.eval()
        with torch.no_grad():
            preds = model(X_test).numpy()
        y_true = y_test.numpy()
        y_pred = (preds >= 0.5).astype(int)

        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, preds))

    # Save performance summary
    suffix = f"_run{run_filter}"
    results = pd.DataFrame({
        "metric": ["accuracy", "f1", "auc"],
        "mean": [np.mean(accs), np.mean(f1s), np.mean(aucs)],
        "std": [np.std(accs), np.std(f1s), np.std(aucs)]
    })
    results.to_csv(f"Results/deep_learning_results{suffix}.csv", index=False)
    print(f"✅ Saved: Results/deep_learning_results{suffix}.csv")
    print(results)

    # Plot learning curve for fold 1
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_fold1, label="Train Loss (Fold 1)")
    plt.plot(val_accuracies_fold1, label="Val Accuracy (Fold 1)")
    plt.xlabel("Epoch")
    plt.title(f"Learning Curve (Run {run_filter}, Fold 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Plots/cnn_learning_curve_run{run_filter}.png", dpi=300)
    plt.clf()
    print(f"📊 Saved: Plots/cnn_learning_curve_run{run_filter}.png")
