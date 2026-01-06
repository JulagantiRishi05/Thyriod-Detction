import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("thyroid.csv")

# Split into features & target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Models
models = {
    "logistic.pkl": LogisticRegression(max_iter=200),
    "random_forest.pkl": RandomForestClassifier(),
    "svm.pkl": SVC(probability=True),
    "knn.pkl": KNeighborsClassifier()
}

accuracies = {}

for file, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[file] = acc
    pickle.dump(model, open(file, "wb"))

print("✅ Models Trained & Saved!")
print(accuracies)
