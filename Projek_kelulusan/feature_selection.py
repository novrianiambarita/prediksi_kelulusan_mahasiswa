import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import joblib

# === 1. LOAD DATASET ===
df = pd.read_csv("dataset_kelulusan_realistic.csv")

# === 2. PISAHKAN FITUR DAN TARGET ===
X = df.drop("target", axis=1)
y = df["target"]

# === 3. NORMALISASI DATA (WAJIB UNTUK CHI2) ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 4. SELEKSI FITUR MENGGUNAKAN CHI-SQUARE ===
selector = SelectKBest(score_func=chi2, k="8")  # k=8 untuk pilih 8 fitur teratas
selector.fit(X_scaled, y)

# === 5. AMBIL FITUR YANG DIPILIH ===
mask = selector.get_support()
selected_features = X.columns[mask].tolist()

# === 6. SIMPAN HASILNYA ===
joblib.dump(selected_features, "selected_features.pkl")

# === 7. CETAK HASIL ===
print("✅ Feature selection selesai.")
print("🎯 Fitur terpilih:", selected_features)
