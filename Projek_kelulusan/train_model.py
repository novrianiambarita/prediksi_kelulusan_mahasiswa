import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# === 1. LOAD DATASET ===
df = pd.read_csv('dataset_kelulusan_realistic.csv')

# === 2. PISAHKAN FITUR DAN TARGET ===
X = df.drop(columns=['target'])
y = df['target']

# === 3. FEATURE SELECTION DENGAN CHI-SQUARE ATAU F-TEST ===
selector = SelectKBest(score_func=f_classif, k=8)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

print("📌 Fitur yang dipilih:", selected_features)

# === 4. STANDARISASI ===
X_selected = X[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# === 5. SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 6. TRAINING LOGISTIC REGRESSION ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === 7. SIMPAN FILE ===
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model, fitur, dan scaler berhasil disimpan.")