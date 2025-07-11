from flask import Flask, request, render_template, redirect, send_file
from datetime import datetime
import pandas as pd
import pickle
import io
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model dan tools
model = pickle.load(open("logistic_model.pkl", "rb"))
selected_features = pickle.load(open("selected_features.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

riwayat_data = []

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

def cek_syarat_manual(data):
    try:
        return (
            float(data["ipk"]) >= 3 and
            int(data["sks"]) >= 144 and
            7 <= int(data["semester"]) <= 14 and
            int(data["kehadiran"]) >= 85 and
            int(data["tidak_lulus"]) == 0
        )
    except:
        return False

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form.to_dict()
    memenuhi_syarat = cek_syarat_manual(input_data)
    final_input, input_dict = encode_input(input_data)

    hasil_prediksi = model.predict(final_input)[0]
    hasil = "Lulus" if hasil_prediksi == 1 and memenuhi_syarat else (
        "Tidak Lulus (Tidak Memenuhi Syarat)" if hasil_prediksi == 1 else "Tidak Lulus"
    )

    input_dict["hasil"] = hasil
    input_dict["jurusan"] = input_data["jurusan"]
    input_dict["waktu"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    riwayat_data.append(input_dict)

    return render_template("index.html", result=hasil, input=input_dict)

@app.route("/riwayat")
def riwayat():
    return render_template("riwayat.html", riwayat=riwayat_data)

@app.route("/delete/<int:index>")
def delete(index):
    if 0 <= index < len(riwayat_data):
        riwayat_data.pop(index)
    return redirect("/riwayat")

@app.route("/edit/<int:index>", methods=["GET", "POST"])
def edit(index):
    if request.method == "GET":
        if 0 <= index < len(riwayat_data):
            return render_template("index.html", edit_data=riwayat_data[index])
        return redirect("/")
    else:
        updated_data = request.form.to_dict()
        _, input_dict = encode_input(updated_data)
        hasil_prediksi = model.predict(_)[0]
        hasil = "Lulus" if hasil_prediksi == 1 and cek_syarat_manual(updated_data) else (
            "Tidak Lulus (Tidak Memenuhi Syarat)" if hasil_prediksi == 1 else "Tidak Lulus"
        )
        input_dict["hasil"] = hasil
        input_dict["jurusan"] = updated_data["jurusan"]
        input_dict["waktu"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 0 <= index < len(riwayat_data):
            riwayat_data[index] = input_dict
        return redirect("/riwayat")

@app.route("/download_csv")
def download_csv():
    df = pd.DataFrame(riwayat_data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="riwayat_prediksi.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
