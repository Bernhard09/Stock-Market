import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime

# --- Konfigurasi Awal ---
today = datetime.date.today()
start = '2010-01-01'
end = today.strftime('%Y-%m-%d')

st.title("Predictive Analysis of Stock Market Trends")

user_input = st.text_input("Enter the Stock Ticker", "AAPL", key="stock_symbol")
df = yf.download(user_input, start=start, end=end, auto_adjust=False)

st.subheader("Data Overview (Recent 15 Days)")
st.dataframe(df.tail(15))

# --- Bagian Visualisasi Moving Average ---
def plot_ma(data, window, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ma = data.rolling(window).mean()
    ax.plot(data, label='Original Close')
    ax.plot(ma, label=f'MA {window}')
    ax.set_title(title)
    ax.legend()
    return fig

col1, col2, col3 = st.columns(3)
if col1.button("Show Closing Prices"):
    fig_all, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.Close)
    st.pyplot(fig_all)

if col2.button("Show MA30"):
    st.pyplot(plot_ma(df.Close, 30, "Closing Price - 30 Days MA"))

if col3.button("Show MA365"):
    st.pyplot(plot_ma(df.Close, 365, "Closing Price - 365 Days MA"))

# --- Load Model ---
# Pastikan file model ada di folder yang sama
try:
    # Ganti 'keras model.h5' menjadi 'keras_model.keras'
    model = load_model("keras_model.keras") 
except Exception as e:
    st.error(f"Model tidak ditemukan atau error: {e}")
    st.stop()

# --- Fungsi Helper Prediksi (Refactored) ---
def predict_next_50_days(series, model, n_steps=100):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 1. Pastikan data di-scale dan di-flatten jadi list angka biasa (1D)
    data_scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))
    # Kita gunakan .flatten() agar isinya [0.1, 0.2, ...] bukan [[0.1], [0.2], ...]
    temp_list = data_scaled[-n_steps:].flatten().tolist() 
    
    predictions = []
    
    for _ in range(50):
        # 2. Ambil 100 data terakhir, ubah format ke (1, 100, 1) sesuai mau model
        x_input = np.array(temp_list[-n_steps:]).reshape(1, n_steps, 1)
        
        # 3. Prediksi (hasilnya biasanya berbentuk array 2D seperti [[0.5]])
        yhat = model.predict(x_input, verbose=0)
        
        # 4. Ambil angkanya saja secara spesifik
        pred_val = float(yhat[0][0])
        
        # 5. Masukkan angkanya ke temp_list untuk bahan prediksi besok
        temp_list.append(pred_val)
        
        # 6. Simpan hasil prediksi untuk di-inverse transform nanti
        predictions.append([pred_val])
        
    return scaler.inverse_transform(predictions)

# --- Proses Prediksi ---
st.subheader("Future Prediction (Next 50 Days)")

# Melakukan prediksi untuk setiap kolom
pred_close = predict_next_50_days(df['Close'], model)
pred_open  = predict_next_50_days(df['Open'], model)
pred_high  = predict_next_50_days(df['High'], model)
pred_low   = predict_next_50_days(df['Low'], model)

# Menggabungkan hasil
result_df = pd.DataFrame({
    'Low': pred_low.flatten(),
    'High': pred_high.flatten(),
    'Open': pred_open.flatten(),
    'Close': pred_close.flatten()
})

# --- Visualisasi Candlestick dengan Matplotlib ---
def plot_matplotlib_candlestick(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tentukan warna (Hijau jika Close > Open, Merah jika sebaliknya)
    colors = ['green' if c >= o else 'red' for c, o in zip(data['Close'], data['Open'])]
    
    # Gambar garis High-Low (Wick)
    ax.vlines(data.index, data['Low'], data['High'], color='black', linewidth=1)
    
    # Gambar badan candle (Open-Close)
    ax.bar(data.index, data['Close'] - data['Open'], bottom=data['Open'], color=colors, width=0.6)
    
    ax.set_title("Predicted Candlestick Chart (Next 50 Days)")
    ax.set_xlabel("Days Forward")
    ax.set_ylabel("Price")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

st.pyplot(plot_matplotlib_candlestick(result_df))
st.write(result_df)

# --- Analisis Return 5 Tahun ---
st.title("Estimated 5-Year Outlook")
try:
    # Pastikan variabel 'end' sudah didefinisikan di awal (today)
    stock_data = yf.download(user_input, start="2023-01-01", end=end, auto_adjust=False)
    
    if not stock_data.empty:
        # Perbaikan: Ambil kolom spesifik sesuai ticker agar menjadi Series tunggal
        # Jika yfinance mengembalikan MultiIndex, kita akses [Level0][Level1]
        if isinstance(stock_data.columns, pd.MultiIndex):
            adj_close_series = stock_data['Adj Close'][user_input]
        else:
            adj_close_series = stock_data['Adj Close']

        price_start = adj_close_series.iloc[0]
        price_end = adj_close_series.iloc[-1]
        
        returns = (price_end / price_start - 1) * 100
        
        # Sekarang 'returns' adalah float murni, aman untuk diformat
        st.write(f"Historical return since 2023 for {user_input}: {returns:.2f}%")
        
        if returns >= 0:
            st.success("Recommendation: YES (Based on current trend)")
        else:
            st.error("Recommendation: NO (Based on current trend)")
    else:
        st.warning("No data found for the selected ticker.")
            
except Exception as e:
    # Ini akan menangkap error dan menampilkannya di UI Streamlit
    st.warning(f"Could not calculate returns: {e}")