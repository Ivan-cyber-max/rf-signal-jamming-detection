import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the exact same 1D-CNN Electronic Warfare Architecture
class RFJammingNet(nn.Module):
    def __init__(self, num_classes=4):
        super(RFJammingNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 2. Load Trained Model Weights
@st.cache_resource
def load_rf_model():
    model = RFJammingNet(num_classes=4)
    model.load_state_dict(torch.load("rf_jamming_classifier.pth", map_location="cpu"))
    model.eval()
    return model

# 3. Live Tactical RF Signal Generator for Real-Time Testing
def simulate_live_capture(signal_type, snr_db=12.0, seq_len=256):
    t = np.linspace(0, 1, seq_len)
    carrier_freq = np.random.uniform(8, 12)
    phase_offset = np.random.uniform(0, 2 * np.pi)
    noise_power = 10 ** (-snr_db / 10.0)
    noise_i = np.random.normal(0, np.sqrt(noise_power / 2), seq_len)
    noise_q = np.random.normal(0, np.sqrt(noise_power / 2), seq_len)

    if signal_type == 0:   # BPSK
        baseband = np.repeat(np.random.choice([-1, 1], size=16), seq_len // 16)
        i_wave = baseband * np.cos(2 * np.pi * carrier_freq * t + phase_offset)
        q_wave = baseband * np.sin(2 * np.pi * carrier_freq * t + phase_offset)
    elif signal_type == 1: # QPSK
        bb_i = np.repeat(np.random.choice([-0.707, 0.707], size=16), seq_len // 16)
        bb_q = np.repeat(np.random.choice([-0.707, 0.707], size=16), seq_len // 16)
        i_wave = bb_i * np.cos(2 * np.pi * carrier_freq * t) - bb_q * np.sin(2 * np.pi * carrier_freq * t)
        q_wave = bb_i * np.sin(2 * np.pi * carrier_freq * t) + bb_q * np.cos(2 * np.pi * carrier_freq * t)
    elif signal_type == 2: # 16-QAM
        levels = [-0.948, -0.316, 0.316, 0.948]
        bb_i = np.repeat(np.random.choice(levels, size=16), seq_len // 16)
        bb_q = np.repeat(np.random.choice(levels, size=16), seq_len // 16)
        i_wave = bb_i * np.cos(2 * np.pi * carrier_freq * t) - bb_q * np.sin(2 * np.pi * carrier_freq * t)
        q_wave = bb_i * np.sin(2 * np.pi * carrier_freq * t) + bb_q * np.cos(2 * np.pi * carrier_freq * t)
    else:                  # HOSTILE JAMMING
        chirp_freq = np.linspace(5, 45, seq_len)
        jam_power = np.random.uniform(1.5, 2.5)
        i_wave = jam_power * np.cos(2 * np.pi * chirp_freq * t) + np.random.normal(0, 0.8, seq_len)
        q_wave = jam_power * np.sin(2 * np.pi * chirp_freq * t) + np.random.normal(0, 0.8, seq_len)

    return np.vstack([i_wave + noise_i, q_wave + noise_q]).astype(np.float32)

# --- STREAMLIT UI DASHBOARD ---
st.set_page_config(page_title="EW Spectrum Sentry", layout="wide")
st.title("📡 Electronic Warfare (EW): RF Signal & Jamming Detection")
st.markdown("**Automated Spectrum Monitoring & Electronic Counter-Countermeasures (ECCM) Decision Support System**")

rf_classes = ['BPSK (Tactical Radio)', 'QPSK (UAV Telemetry)', '16-QAM (High-Speed Link)', 'HOSTILE JAMMING']

st.sidebar.header("🎛️ Tactical Spectrum Simulator")
selected_signal = st.sidebar.selectbox(
    "Inject Incoming RF Transmission:",
    options=[0, 1, 2, 3],
    format_func=lambda x: rf_classes[x]
)
snr_slider = st.sidebar.slider("Channel Signal-to-Noise Ratio (SNR dB)", min_value=5, max_value=20, value=12)
st.sidebar.button("⚡ Capture & Analyze Live RF Burst", use_container_width=True)

# Generate and analyze signal
iq_sample = simulate_live_capture(selected_signal, snr_db=snr_slider)
model = load_rf_model()

with torch.no_grad():
    input_tensor = torch.tensor(iq_sample).unsqueeze(0)
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)

pred_class = int(pred_idx.item())
confidence_pct = float(conf.item()) * 100.0

# Display Tactical Alert Banner
if pred_class == 3:
    st.error(f"### 🚨 ALERT: **{rf_classes[pred_class]} DETECTED!** (Confidence: **{confidence_pct:.2f}%**)")
    st.warning("⚠️ **Recommended Action:** Initiate Frequency-Hopping Spread Spectrum (FHSS) channel switch and alert Electronic Warfare Officer.")
else:
    st.success(f"### ✅ Verified Friendly Link: **{rf_classes[pred_class]}** (Confidence: **{confidence_pct:.2f}%**)")

# Plot Time-Domain I/Q Waveforms & Frequency Power Spectrum Side-by-Side
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Time-Domain I/Q Waveform")
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(iq_sample[0][:120], label="In-Phase (I)", color="navy")
    ax1.plot(iq_sample[1][:120], label="Quadrature (Q)", color="darkorange", alpha=0.85)
    ax1.set_xlabel("Time Samples")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig1)

with col2:
    st.subheader("2. FFT Frequency Power Spectrum")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    complex_sig = iq_sample[0] + 1j * iq_sample[1]
    fft_vals = np.fft.fftshift(np.fft.fft(complex_sig))
    power_db = 20 * np.log10(np.abs(fft_vals) + 1e-6)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_sig)))
    ax2.plot(freqs, power_db, color="crimson" if pred_class == 3 else "teal", linewidth=1.5)
    ax2.set_xlabel("Normalized Frequency Offset")
    ax2.set_ylabel("Power Spectral Density (dB)")
    ax2.set_ylim(-10, 65)
    ax2.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig2)
