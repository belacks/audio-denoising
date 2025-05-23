import streamlit as st
import numpy as np
import torch
from torch import nn
import torchaudio
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
import os
from io import BytesIO 
import time
import av # Ditambahkan untuk AudioFrame

# --- Konfigurasi Model ---
MODEL_PATH = os.path.join("saves", "GRUUNet2-dari_tult2", "checkpoint.pth")
TARGET_SR = 48000
MODEL_EXPECTS_CHANNELS = 1

# --- Konfigurasi GRUUNet2 (dari main.ipynb Anda) ---
GRUUNET2_CONFIG = {
    "num_compressed_bins": 4,
    "in_size": MODEL_EXPECTS_CHANNELS,
    "hidden_sizes": [17, 17, 17, 17],
    "kernel_sizes": [3, 3, 3, 3],
    "strides": [2, 2, 2, 2],
    "paddings": [1, 1, 1, 1],
    "num_gaussians": 6
}

# --- Parameter STFT/Mel ---
STFT_PARAMS = {
    "n_fft": 1536, 
    "hop_length": 768, 
    "n_mels": 64
}

# --- Impor Model dari gruunet2.py ---
GRUUNet2_class = None
try:
    from gruunet2 import GRUUNet2
    GRUUNet2_class = GRUUNet2
except ImportError:
    pass
except Exception as e:
    pass

# --- Fungsi untuk Memuat Model PyTorch ---
@st.cache_resource
def load_denoising_model_pytorch(model_path, ModelClass_to_load, correct_config):
    if ModelClass_to_load is None:
        return None, None

    model_config_to_use = None
    state_dict_to_load = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if not os.path.exists(model_path):
            return None, None

        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            if 'hparams' in checkpoint and isinstance(checkpoint['hparams'], dict):
                model_config_to_use = checkpoint['hparams']
            elif 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                model_config_to_use = checkpoint['config']

            if 'model_state_dict' in checkpoint:
                state_dict_to_load = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict_to_load = checkpoint['state_dict']
            else: 
                potential_sd = {k: v for k, v in checkpoint.items() if k not in ['hparams', 'config', 'last_epoch']} 
                if potential_sd and all(isinstance(v, torch.Tensor) for v in potential_sd.values()):
                    state_dict_to_load = potential_sd
        elif hasattr(checkpoint, 'state_dict') and callable(checkpoint.state_dict): 
             state_dict_to_load = checkpoint.state_dict()
             if hasattr(checkpoint, 'hparams') and isinstance(checkpoint.hparams, dict): model_config_to_use = checkpoint.hparams
             elif hasattr(checkpoint, 'config') and isinstance(checkpoint.config, dict): model_config_to_use = checkpoint.config
        elif isinstance(checkpoint, (dict, torch.Tensor)) and all(isinstance(v, torch.Tensor) for v in (checkpoint.values() if isinstance(checkpoint, dict) else [checkpoint] if isinstance(checkpoint, torch.Tensor) else [])):
             state_dict_to_load = checkpoint


        if state_dict_to_load is None:
            return None, None

        if model_config_to_use is None:
            model_config_to_use = correct_config
        
        known_gruunet2_params_names = [
            "num_compressed_bins", "in_size", "hidden_sizes",
            "kernel_sizes", "strides", "paddings", "num_gaussians"
        ]
        
        final_init_args = {}
        for param_name in known_gruunet2_params_names:
            if param_name in model_config_to_use:
                final_init_args[param_name] = model_config_to_use[param_name]
        
        required_params_for_gruunet2 = [
            "num_compressed_bins", "in_size", "hidden_sizes",
            "kernel_sizes", "strides", "paddings"
        ]
        
        missing_required_params = []
        for req_param in required_params_for_gruunet2:
            if req_param not in final_init_args:
                missing_required_params.append(req_param)
        
        if missing_required_params:
            return None, None
            
        model_architecture = ModelClass_to_load(**final_init_args)
        model_architecture.load_state_dict(state_dict_to_load)
        model_architecture.eval()
        model_architecture.to(device)
        return model_architecture, device

    except Exception as e:
        return None, None


# --- Kelas Audio Processor untuk streamlit-webrtc ---
class DenoisingAudioProcessor(AudioProcessorBase):
    def __init__(self, model, device, gru_config, stft_params, target_sr):
        self.model = model
        self.device = device
        self.gru_config = gru_config
        self.stft_params = stft_params 
        self.target_sr = target_sr
        self.hx = None 
        
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_ola_buffer = np.zeros(self.stft_params["n_fft"], dtype=np.float32) 

        self.T0_transform = torchaudio.transforms.Spectrogram(
            power=None, n_fft=self.stft_params['n_fft'],
            win_length=self.stft_params['n_fft'], hop_length=self.stft_params['hop_length'],
            window_fn=torch.hann_window 
        ).to(device)
        self.M0T_transform = torchaudio.transforms.MelScale(
            n_mels=self.stft_params['n_mels'], n_stft=self.stft_params['n_fft'] // 2 + 1,
            sample_rate=target_sr
        ).to(device)
        
        self.M0I_transform = torchaudio.transforms.InverseMelScale(
            n_mels=self.stft_params['n_mels'], n_stft=self.stft_params['n_fft'] // 2 + 1,
            sample_rate=target_sr
        ).to(device)
        self.griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=self.stft_params['n_fft'], win_length=self.stft_params['n_fft'],
            hop_length=self.stft_params['hop_length'], window_fn=torch.hann_window,
            power=1.0 
        ).to(device) 

        self.hann_window_np = torch.hann_window(self.stft_params['n_fft']).cpu().numpy()


    def _initialize_hx(self, batch_size, dtype):
        self.hx = torch.zeros(
            batch_size,
            self.gru_config["hidden_sizes"][-1], 
            self.gru_config["num_compressed_bins"],
            dtype=dtype,
            device=self.device,
        )

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame: # Menambahkan type hint untuk frame
        in_data_s16 = frame.to_ndarray(format="s16") # Layout C adalah default
        if in_data_s16.ndim > 1: 
            in_data_s16 = in_data_s16[:,0]

        audio_chunk_float32 = (in_data_s16.astype(np.float32) / np.iinfo(np.int16).max)
        
        self.input_buffer = np.concatenate([self.input_buffer, audio_chunk_float32])
        
        processed_output_for_this_recv = np.array([], dtype=np.float32)

        while len(self.input_buffer) >= self.stft_params["n_fft"]:
            current_frame_input = self.input_buffer[:self.stft_params["n_fft"]]
            
            peak = np.max(np.abs(current_frame_input))
            if peak > 1e-6:
                frame_normalized = current_frame_input / peak
            else:
                frame_normalized = current_frame_input
                peak = 1.0 

            windowed_frame = frame_normalized * self.hann_window_np
            waveform_tensor = torch.from_numpy(windowed_frame).float().unsqueeze(0).to(self.device)

            complex_spec = self.T0_transform(waveform_tensor) 
            mag_spec = complex_spec.abs()
            mel_logmag_spec = self.M0T_transform(mag_spec).log1p() 
            
            model_input = mel_logmag_spec.transpose(-1, -2)

            if self.hx is None or self.hx.shape[0] != model_input.shape[0] or self.hx.dtype != model_input.dtype:
                self._initialize_hx(model_input.shape[0], model_input.dtype)

            with torch.no_grad():
                predicted_diff_mel, self.hx = self.model(model_input, self.hx)
            
            reconstructed_signal_mel = model_input - predicted_diff_mel
            reconstructed_signal_mel = nn.functional.leaky_relu(reconstructed_signal_mel, negative_slope=0.2)

            reconstructed_signal_mel_T = reconstructed_signal_mel.transpose(-1, -2)
            reconstructed_mag_mel = torch.expm1(reconstructed_signal_mel_T) 
            reconstructed_mag_mel = torch.clamp(reconstructed_mag_mel, min=0) 
            
            reconstructed_linear_mag_spec = self.M0I_transform(reconstructed_mag_mel)
            reconstructed_linear_mag_spec = torch.clamp(reconstructed_linear_mag_spec, min=0)
            
            denoised_frame_tensor = self.griffin_lim_transform(reconstructed_linear_mag_spec)
            
            denoised_frame_np_raw = denoised_frame_tensor.squeeze(0).cpu().numpy()
            denoised_frame_np_windowed = denoised_frame_np_raw 
            denoised_frame_np_denormalized = denoised_frame_np_windowed * peak 
            
            current_output_segment = self.output_ola_buffer[:self.stft_params["hop_length"]].copy()
            processed_output_for_this_recv = np.concatenate([processed_output_for_this_recv, current_output_segment])
            
            self.output_ola_buffer[:-self.stft_params["hop_length"]] = self.output_ola_buffer[self.stft_params["hop_length"]:]
            self.output_ola_buffer[-self.stft_params["hop_length"]:] = 0.0 
            self.output_ola_buffer[:self.stft_params["n_fft"]] += denoised_frame_np_denormalized 
            
            self.input_buffer = self.input_buffer[self.stft_params["hop_length"]:]
        
        if len(processed_output_for_this_recv) == 0:
             # Menggunakan av.AudioFrame.from_ndarray untuk membuat frame passthrough/silence
             # dengan format yang benar (s16) dan panjang yang sama dengan frame input asli
             # Ini untuk menjaga konsistensi dan menghindari error jika tidak ada data yang diproses
             out_data_s16_passthrough = (np.clip(audio_chunk_float32, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
             
             # Pastikan panjangnya sesuai dengan frame.samples jika audio_chunk_float32 lebih pendek
             if len(out_data_s16_passthrough) < frame.samples:
                 padding = np.zeros(frame.samples - len(out_data_s16_passthrough), dtype=np.int16)
                 out_data_s16_passthrough = np.concatenate([out_data_s16_passthrough, padding])
             elif len(out_data_s16_passthrough) > frame.samples:
                 out_data_s16_passthrough = out_data_s16_passthrough[:frame.samples]

             return av.AudioFrame.from_ndarray(out_data_s16_passthrough.reshape(1, -1), format="s16", layout="mono")


        out_audio_chunk_clipped = np.clip(processed_output_for_this_recv, -1.0, 1.0)
        out_data_s16 = (out_audio_chunk_clipped * np.iinfo(np.int16).max).astype(np.int16)
        
        # Menggunakan av.AudioFrame.from_ndarray untuk membuat frame output
        # Pastikan ndarray adalah 2D (channels, samples) untuk format="s16", layout="mono" atau "stereo"
        # Karena kita selalu mono di sini:
        return av.AudioFrame.from_ndarray(out_data_s16.reshape(1, -1), format="s16", layout="mono")


# --- UI Utama Streamlit ---

st.set_page_config(
    page_title="Suara Jernih Real-time",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom untuk mempercantik tampilan
st.markdown("""
    <style>
        /* Latar Belakang Utama */
        .stApp {
            background: linear-gradient(to bottom right, #1F2937, #111827); /* Dark gray gradient */
            color: #F9FAFB; /* Light text */
        }

        /* Sidebar Styling */
        .css-1d391kg { /* Kelas CSS untuk sidebar Streamlit */
            background-color: #1F2937; /* Darker sidebar */
            border-right: 1px solid #374151;
        }
        .css-1d391kg .st-emotion-cache-10oheav { /* Teks di sidebar */
            color: #D1D5DB;
        }
        .css-1d391kg .st-emotion-cache-6q9sum { /* Header di sidebar */
            color: #9CA3AF;
            font-weight: bold;
        }

        /* Tombol Utama */
        .stButton>button {
            border: 2px solid #4F46E5; /* Indigo border */
            border-radius: 20px;
            padding: 10px 24px;
            color: white;
            background-color: #4F46E5; /* Indigo background */
            transition: all 0.3s ease-in-out;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #6366F1; /* Lighter Indigo */
            border-color: #6366F1;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        .stButton>button:disabled {
            background-color: #4B5563; /* Gray when disabled */
            border-color: #4B5563;
            color: #9CA3AF;
            opacity: 0.7;
        }
        
        /* Kontainer Utama */
        .main-container {
            padding: 2rem;
            background-color: #111827; /* Slightly lighter than app background */
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }

        /* Judul Aplikasi */
        .app-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ECFDF5; /* Mint green */
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .app-subtitle {
            font-size: 1.1rem;
            color: #9CA3AF; /* Gray text */
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Pesan Status */
        .stAlert {
            border-radius: 8px;
            border-left-width: 5px;
        }
        .stAlert.st-emotion-cache-j6npxx.e1nzilvr5 { /* Success */
             border-left-color: #34D399 !important; /* Green */
        }
        .stAlert.st-emotion-cache-j6npxx.e1nzilvr3 { /* Info */
             border-left-color: #60A5FA !important; /* Blue */
        }
        .stAlert.st-emotion-cache-j6npxx.e1nzilvr4 { /* Warning */
             border-left-color: #FBBF24 !important; /* Yellow */
        }
        .stAlert.st-emotion-cache-j6npxx.e1nzilvr2 { /* Error */
             border-left-color: #F87171 !important; /* Red */
        }

        /* Styling untuk webrtc_streamer */
        div[data-testid="stVideo"] { 
            display: none; 
        }
        
        .webrtc-container {
            background-color: #1F2937; 
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #374151; /* Border lebih halus */
            box-shadow: 0 8px 16px rgba(0,0,0,0.4); /* Bayangan lebih dalam */
            margin-top: 1rem;
            margin-bottom: 1.5rem; /* Spasi lebih di bawah */
            transition: all 0.3s ease-in-out;
        }
        .webrtc-container:hover {
            border-color: #4F46E5; /* Highlight border on hover */
        }

        /* Tombol internal webrtc_streamer (SELECT DEVICE, START) */
        /* Ini mungkin tidak selalu berhasil karena enkapsulasi komponen */
        .webrtc-container div[data-testid="stToolbar"] button {
            background-color: #374151 !important;
            border: 1px solid #4B5563 !important;
            color: #D1D5DB !important;
            font-weight: 500 !important; /* Sedikit lebih tebal */
            padding: 8px 16px !important;
            border-radius: 8px !important;
            transition: background-color 0.2s ease, border-color 0.2s ease;
        }
        .webrtc-container div[data-testid="stToolbar"] button:hover {
            background-color: #4B5563 !important;
            border-color: #6B7280 !important;
        }
         .webrtc-container div[data-testid="stToolbar"] label { /* Label "START", "STOP" */
            color: #9CA3AF !important;
        }


        .webrtc-container p { 
            color: #9CA3AF;
            font-style: italic;
            text-align: center;
            margin-top: 0.5rem; /* Spasi di atas teks status */
        }
        
        .webrtc-instruction {
            font-size: 0.95rem;
            color: #A0AEC0; /* Slightly lighter gray */
            text-align: center;
            margin-bottom: 0.75rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }

        audio { 
            width: 100%;
            margin-top: 1rem;
            border-radius: 8px;
            border: 1px solid #374151;
            background-color: #0F172A; 
        }
        audio::-webkit-media-controls-panel {
            background-color: #1E293B;
            color: #94A3B8;
        }
        audio::-webkit-media-controls-play-button,
        audio::-webkit-media-controls-volume-slider,
        audio::-webkit-media-controls-mute-button,
        audio::-webkit-media-controls-timeline {
            filter: invert(70%) sepia(10%) saturate(500%) hue-rotate(180deg) brightness(90%) contrast(90%);
        }


    </style>
""", unsafe_allow_html=True)


# --- UI Utama Streamlit ---

st.markdown("<p class='app-title'>🎙️ Penjernih Suara Real-time</p>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Hilangkan noise dari audio Anda secara langsung, seperti fitur Discord!</p>", unsafe_allow_html=True)

with st.container():
    if GRUUNet2_class is None:
        st.sidebar.error(
            "Gagal mengimpor definisi model 'GRUUNet2' dari 'gruunet2.py'. "
            "Pastikan file tersebut ada dan benar."
        )
        try:
            from gruunet2 import GRUUNet2 
        except Exception as import_err:
            st.sidebar.expander("Detail Error Impor").error(import_err)
        st.error("Definisi model tidak ditemukan. Aplikasi tidak dapat berjalan.")
        st.stop() 
    else:
        st.sidebar.success("Definisi model 'GRUUNet2' berhasil diimpor.")

    model, device = load_denoising_model_pytorch(MODEL_PATH, GRUUNet2_class, GRUUNET2_CONFIG)

    if model is None:
        st.error(f"Gagal memuat model dari '{MODEL_PATH}'. Aplikasi tidak dapat berjalan.")
        if not os.path.exists(MODEL_PATH):
             st.info(f"File model tidak ditemukan pada path yang diharapkan: {os.path.abspath(MODEL_PATH)}")
        st.stop()
    else:
        st.success(f"✅ Model '{os.path.basename(MODEL_PATH)}' berhasil dimuat ke **{device}**.")


    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    if 'denoising_active' not in st.session_state:
        st.session_state.denoising_active = False
    if 'webrtc_key' not in st.session_state:
        st.session_state.webrtc_key = "denoising-stream-initial"


    st.markdown("---")
    cols_button = st.columns([1, 0.2, 1]) 
    with cols_button[0]:
        if st.button("🚀 Mulai Penjernihan Suara", type="primary", disabled=st.session_state.denoising_active, use_container_width=True):
            st.session_state.denoising_active = True
            st.session_state.webrtc_key = f"denoising-stream-{time.time()}" 
            st.rerun() 

    with cols_button[2]:
        if st.button(" Hentikan Penjernihan", disabled=not st.session_state.denoising_active, use_container_width=True):
            st.session_state.denoising_active = False
            st.rerun() 
    
    st.markdown("---")


    if st.session_state.denoising_active:
        st.info("🎧 Penjernihan suara aktif. Silakan berbicara ke mikrofon Anda.")
        
        # Instruksi pemilihan perangkat
        st.markdown(
            "<p class='webrtc-instruction'>Untuk memilih <b>mikrofon input</b> atau <b>speaker output</b> yang berbeda, "
            "gunakan tombol <b>'SELECT DEVICE'</b> yang muncul di bawah ini (biasanya sebelum streaming dimulai atau saat izin mikrofon diminta oleh browser Anda). "
            "Pilihan output speaker juga dipengaruhi oleh pengaturan default sistem Anda.</p>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='webrtc-container'>", unsafe_allow_html=True)
        
        audio_processor_factory = lambda: DenoisingAudioProcessor(
            model=model, 
            device=device, 
            gru_config=GRUUNET2_CONFIG, 
            stft_params=STFT_PARAMS, 
            target_sr=TARGET_SR
        )

        webrtc_ctx = webrtc_streamer(
            key=st.session_state.webrtc_key, 
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            audio_processor_factory=audio_processor_factory,
            media_stream_constraints={
                "audio": {
                    "sampleRate": TARGET_SR, 
                    "channelCount": 1      
                }, 
                "video": False 
            },
            async_processing=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


        if webrtc_ctx.state.playing:
            st.success("🎤 Streaming audio aktif dan suara sedang dijernihkan...")
        elif not webrtc_ctx.state.playing and st.session_state.denoising_active : 
             st.warning("Menunggu koneksi atau izin mikrofon...")


    else:
        st.info("Tekan 'Mulai Penjernihan Suara' untuk memulai.")
    
# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan & Info")
with st.sidebar.expander("Konfigurasi Model (Read-only)", expanded=False):
    st.json({"Model Path": MODEL_PATH, "Target SR": TARGET_SR})
    st.json({"STFT Parameters": STFT_PARAMS})
    st.json({"GRUUNet2 Config": GRUUNET2_CONFIG})

with st.sidebar.expander("Path Debug (Read-only)", expanded=False):
    st.text(f"Direktori kerja: {os.getcwd()}")
    st.text(f"Path model dicari: {os.path.abspath(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        st.success("File model ditemukan.")
    else:
        st.error("File model TIDAK ditemukan.")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan model deep learning GRUUNet2 (PyTorch) "
    "untuk mereduksi noise dari audio secara real-time. "
    "Dibuat dengan Streamlit dan PyTorch."
)
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <p style="font-size: 0.9em; color: #9CA3AF;">
            Versi Aplikasi: 1.2.0 <br> 
            Dikembangkan untuk Proyek Penprop
        </p>
    </div>
    """, unsafe_allow_html=True
)