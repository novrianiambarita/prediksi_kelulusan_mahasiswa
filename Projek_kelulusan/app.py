import os
import pickle
from flask import Flask, request, render_template, redirect, url_for, send_file
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Path absolut agar aman di PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dan fitur
model = pickle.load(open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb"))
selected_features = pickle.load(open(os.path.join(BASE_DIR, "selected_features.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# Riwayat prediksi disimpan di CSV
RIWAYAT_FILE = os.path.join(BASE_DIR, "riwayat_prediksi.csv")

# Load riwayat jika file sudah ada
if os.path.exists(RIWAYAT_FILE):
    riwayat_df = pd.read_csv(RIWAYAT_FILE)
else:
    riwayat_df = pd.DataFrame()

# ------------- Encode Input -------------
def encode_input(data):
    jurusan_dict = {
        "jurusan_Akuntansi": 0,
        "jurusan_Manajemen": 0,
        "jurusan_Sistem Informasi": 0,
        "jurusan_Teknik Informatika": 0,
    }
    jurusan_key = f"jurusan_{data['jurusan']}"
    if jurusan_key in jurusan_dict:
        jurusan_dict[jurusan_key] = 1

    input_dict = {
        "ipk": float(data["ipk"]),
        "sks": int(data["sks"]),
        "kehadiran": int(data["kehadiran"]),
        "tidak_lulus": int(data["tidak_lulus"]),
        "organisasi": int(data["organisasi"]),
        "semester": int(data["semester"]),
        **jurusan_dict,
    }

    df_input = pd.DataFrame([input_dict])
    df_selected = df_input[selected_features]
    df_scaled = scaler.transform(df_selected)
    return df_scaled, input_dict

# ------------- Validasi Manual Kelulusan -------------
def cek_syarat_manual(data):
    try:
        return (
            float(data['ipk']) >= 3 and
            int(data['sks']) >= 144 and
            7 <= int(data['semester']) <= 14 and
            int(data['kehadiran']) >= 85 and
            int(data['tidak_lulus']) == 0
        )
    except:
        return False

# ------------- Halaman Utama -------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = request.form.to_dict()
        memenuhi_syarat = cek_syarat_manual(input_data)
        final_input, input_dict = encode_input(input_data)
        hasil_prediksi = model.predict(final_input)[0]

        if hasil_prediksi == 1 and memenuhi_syarat:
            hasil = "Lulus"
        elif hasil_prediksi == 1 and not memenuhi_syarat:
            hasil = "Tidak Lulus (Tidak Memenuhi Syarat)"
        else:
            hasil = "Tidak Lulus"

        input_dict["hasil"] = hasil
        input_dict["waktu"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        global riwayat_df
        riwayat_df = pd.concat([riwayat_df, pd.DataFrame([input_dict])], ignore_index=True)
        riwayat_df.to_csv(RIWAYAT_FILE, index=False)

        return render_template("index.html", result=hasil, input=input_dict)

    return render_template("index.html")

# ------------- Halaman Riwayat -------------
@app.route("/riwayat")
def riwayat():
    if riwayat_df.empty:
        return render_template("riwayat.html", riwayat=[])
    return render_template("riwayat.html", riwayat=riwayat_df.to_dict(orient="records"))

# ------------- Unduh CSV -------------
@app.route("/unduh")
def unduh_csv():
    return send_file(RIWAYAT_FILE, as_attachment=True)

# ------------- Hapus Riwayat Baris -------------
@app.route("/hapus/<int:index>")
def hapus(index):
    global riwayat_df
    if 0 <= index < len(riwayat_df):
        riwayat_df = riwayat_df.drop(index).reset_index(drop=True)
        riwayat_df.to_csv(RIWAYAT_FILE, index=False)
    return redirect(url_for("riwayat"))

# ------------- Edit Riwayat -------------
@app.route("/edit/<int:index>", methods=["GET", "POST"])
def edit(index):
    global riwayat_df
    if request.method == "POST":
        for col in request.form:
            riwayat_df.at[index, col] = request.form[col]
        riwayat_df.to_csv(RIWAYAT_FILE, index=False)
        return redirect(url_for("riwayat"))

    row_data = riwayat_df.loc[index].to_dict()
    return render_template("edit.html", index=index, data=row_data)

# ------------- Menjalankan Aplikasi -------------
if __name__ == "__main__":
    app.run(debug=True)
