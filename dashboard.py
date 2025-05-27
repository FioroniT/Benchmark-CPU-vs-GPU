import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import platform
import subprocess
import os

# Configuración inicial
st.set_page_config(page_title="Benchmark CPU vs GPU", layout="wide")
st.title("⚙️ Benchmark de Entrenamiento - TensorFlow")

# Mostrar información del sistema
st.subheader("🖥️ Información del Sistema")

def get_cpu_name():
    if platform.system() == "Windows":
        try:
            command = 'powershell "Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name"'
            output = subprocess.check_output(command, shell=True, encoding='utf-8', errors='ignore')
            return output.strip()
        except Exception:
            return platform.processor()
    return platform.processor()
cpu_name = get_cpu_name()
st.markdown(f"**Procesador:** {cpu_name}")

def get_gpu_name():
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output("dxdiag /t dxdiag_output.txt", shell=True)
            with open("dxdiag_output.txt", encoding='utf-8', errors='ignore') as f:
                content = f.read()
            os.remove("dxdiag_output.txt")
            for line in content.splitlines():
                if "Card name:" in line:
                    return line.split("Card name:")[1].strip()
        except Exception:
            return "GPU detectada pero no se pudo obtener el nombre"
    return "No detectada"

# Detectar GPU
gpu_name = "No detectada"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        gpu_name = get_gpu_name()
    except Exception as e:
        st.warning(f"No se pudo configurar la GPU: {e}")

st.markdown(f"**GPU:** {gpu_name}")

# Datos globales
X = np.random.randn(10000, 1000).astype(np.float32)
y = np.random.randint(0, 10, size=(10000,))

def create_model(hidden_units=500):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(10)
    ])

def train(device_name, epochs=100, batch_size=64, hidden_units=500):
    with tf.device(device_name):
        model = create_model(hidden_units)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        times = []
        for epoch in range(epochs):
            start = time.time()
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)
            end = time.time()
            times.append(end - start)
        total = sum(times)
        average = total / epochs
        return total, average

def run_benchmark(epochs, batch_size, hidden_units):
    results = []
    total_cpu, avg_cpu = train('/CPU:0', epochs, batch_size, hidden_units)
    results.append((cpu_name, total_cpu, avg_cpu))

    if gpus:
        try:
            total_gpu, avg_gpu = train('/GPU:0', epochs, batch_size, hidden_units)
            results.append((gpu_name, total_gpu, avg_gpu))
        except Exception as e:
            st.warning(f"No se pudo ejecutar en GPU: {e}")

    return results

# Sidebar de control
with st.sidebar:
    st.header("Parámetros del Benchmark")
    epochs = st.slider("Epochs", min_value=1, max_value=200, value=100, step=10)
    batch_size = st.slider("Batch Size", min_value=16, max_value=1024, value=64, step=16)
    hidden_units = st.slider("Unidades ocultas", min_value=100, max_value=2000, value=500, step=100)

    if st.button("Ejecutar Benchmark"):
        with st.spinner("Ejecutando entrenamiento..."):
            results = run_benchmark(epochs, batch_size, hidden_units)
            st.session_state["benchmark_results"] = pd.DataFrame(
                results,
                columns=['Dispositivo', 'Tiempo total (s)', 'Tiempo promedio por epoch (s)']
            )
            st.success("Benchmark completado")

# Mostrar resultados si ya se ejecutó el benchmark
if "benchmark_results" in st.session_state:
    df = st.session_state["benchmark_results"]
    st.subheader("📊 Resultados del Benchmark")
    st.dataframe(df.set_index('Dispositivo'))

    fig, ax = plt.subplots()
    x = range(len(df))
    width = 0.35
    ax.bar(x, df['Tiempo total (s)'], width, label='Total (s)')
    ax.bar([i + width for i in x], df['Tiempo promedio por epoch (s)'], width, label='Promedio por Epoch (s)')
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(df['Dispositivo'])
    ax.set_title("Rendimiento CPU vs GPU")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Configura los parámetros y presiona el botón para ejecutar el benchmark.")
