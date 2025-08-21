import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
df = pd.read_csv(r"D:\StudentsPerformance.csv")
print(df.head())
df["pass_math"] = (df["math score"] >= 70).astype(int)
X = df.drop(["math score", "pass_math"], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df["pass_math"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
print("Training data shape:", X_train_tensor.shape)
print("Test data shape:", X_test_tensor.shape)

import torch.nn as nn
import torch.optim as optim

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        preds = (outputs >= 0.5).float()
        acc = (preds.eq(y_train_tensor).sum() / y_train_tensor.shape[0]).item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")
with torch.no_grad():
    test_preds = (model(X_test_tensor) >= 0.5).float()
    test_acc = (test_preds.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
print("Test Accuracy:", test_acc)

model_reg = LogisticRegressionModel(input_dim)
optimizer_reg = optim.SGD(model_reg.parameters(), lr=0.01, weight_decay=0.01)
for epoch in range(100):
    outputs = model_reg(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer_reg.zero_grad()
    loss.backward()
    optimizer_reg.step()
with torch.no_grad():
    preds = (model_reg(X_test_tensor) >= 0.5).float()
    acc = (preds.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
print("Test Accuracy with L2 Regularization:", acc)



torch.save(model.state_dict(), "students_logreg.pth")
loaded_model = LogisticRegressionModel(input_dim)
loaded_model.load_state_dict(torch.load("students_logreg.pth"))
loaded_model.eval()
with torch.no_grad():
    preds_loaded = (loaded_model(X_test_tensor) >= 0.5).float()
    acc_loaded = (preds_loaded.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
print("Accuracy after loading model:", acc_loaded)

learning_rates = [0.001, 0.01, 0.1]
best_acc = 0
best_lr = None

for lr in learning_rates:
    model_tune = LogisticRegressionModel(input_dim)
    optimizer_tune = optim.SGD(model_tune.parameters(), lr=lr)

    for epoch in range(50):
        outputs = model_tune(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer_tune.zero_grad()
        loss.backward()
        optimizer_tune.step()

    with torch.no_grad():
        preds = (model_tune(X_test_tensor) >= 0.5).float()
        acc = (preds.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()

    print(f"LR={lr}, Test Acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_lr = lr

print("Best Learning Rate:", best_lr, "with Accuracy:", best_acc)

weights = model.linear.weight.detach().numpy().flatten()
features = X.columns

importance_df = pd.DataFrame({"Feature": features, "Importance": weights})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print(importance_df)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Weight (Importance)")
plt.title("Feature Importance - Logistic Regression")
plt.gca().invert_yaxis()
plt.show()
