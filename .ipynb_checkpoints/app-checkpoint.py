# app_streamlit.py (atau app_streamlit_pytorch.py)

import streamlit as st # Streamlit HARUS diimpor sebelum pemanggilan st.set_page_config
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
import torch
import os # Untuk path relatif

# --- Antarmuka Streamlit - Konfigurasi Halaman HARUS PERTAMA ---
# Pastikan ini adalah perintah Streamlit PERTAMA yang dieksekusi
st.set_page_config(page_title="Audio Denoising (PyTorch)", layout="wide")

# --- Konfigurasi Model ---
# Sesuaikan path ini dengan lokasi model Anda relatif terhadap app_streamlit.py
# Jika app_streamlit.py ada di 'audio_denoising-master', maka pathnya:
MODEL_PATH = os.path.join("saves", "GRUUNet2-dari_tult", "checkpoint.pth")

# Sesuaikan dengan sample rate yang diharapkan model Anda (dari main.ipynb atau script training)
TARGET_SR = 48000  # Ganti jika berbeda
# Sesuaikan dengan ekspektasi channel model (biasanya 1 untuk mono)
MODEL_EXPECTS_CHANNELS = 1 # Ganti jika model Anda multi-channel

# --- Impor Model dan Fungsi dari Proyek Anda ---
# Anda perlu mengadaptasi bagian ini berdasarkan struktur kode Anda.
# Misalnya, jika model didefinisikan di gruunet2.py:
GRUUnet2_model_class = None # Inisialisasi
try:
    # Pastikan file gruunet2.py ada di direktori yang sama atau PYTHONPATH
    # dan berisi definisi kelas model PyTorch Anda (misalnya GRUUnet2_model)
    from gruunet2 import GRUUNet2 # GANTI NAMA KELAS JIKA BERBEDA
    GRUUnet2_model_class = GRUUNet2 # Simpan kelas model yang berhasil diimpor
    # Pindahkan st.sidebar.success ke bagian utama aplikasi setelah st.set_page_config
except ImportError:
    # Pindahkan st.error ke bagian utama aplikasi
    pass # Penanganan error akan dilakukan di bagian utama UI
except Exception as e:
    # Pindahkan st.error ke bagian utama UI
    pass # Penanganan error akan dilakukan di bagian utama UI


# --- Fungsi untuk Memuat Model PyTorch ---
@st.cache_resource # Cache model agar tidak di-load ulang setiap interaksi
def load_denoising_model_pytorch(model_path, ModelClass):
    """
    Muat model denoising PyTorch Anda.
    """
    if ModelClass is None:
        # st.error dipanggil di UI utama jika ModelClass adalah None
        return None
    try:
        # 1. Inisialisasi arsitektur model
        # Anda HARUS mengetahui parameter yang dibutuhkan untuk inisialisasi kelas ModelClass.
        model_architecture = ModelClass() # Sesuaikan jika ada argumen!

        # 2. Muat state dictionary dari file .pth
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            # st.error dan st.info dipanggil di UI utama
            return None

        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model_architecture.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model_architecture.load_state_dict(checkpoint['state_dict'])
        else: 
            model_architecture.load_state_dict(checkpoint)

        model_architecture.eval()
        model_architecture.to(device) 

        # st.success dipanggil di UI utama
        return model_architecture, device
    except FileNotFoundError:
        # st.error dipanggil di UI utama
        return None, None
    except AttributeError as e:
        # st.error dipanggil di UI utama
        return None, None
    except Exception as e:
        # st.error dan traceback dipanggil di UI utama
        return None, None

# --- Fungsi untuk Memproses Audio dengan PyTorch ---
def process_audio_pytorch(audio_data_input, sr_input, model_loaded, device):
    """
    Preprocess, denoise (PyTorch), dan postprocess audio.
    """
    if model_loaded is None:
        return audio_data_input, sr_input 

    audio_data = audio_data_input.copy()

    if sr_input != TARGET_SR:
        audio_data = librosa.resample(audio_data, orig_sr=sr_input, target_sr=TARGET_SR)
    current_sr = TARGET_SR

    if audio_data.ndim > 1 and audio_data.shape[-1] != MODEL_EXPECTS_CHANNELS:
        if MODEL_EXPECTS_CHANNELS == 1:
            audio_data = librosa.to_mono(audio_data)
    elif audio_data.ndim == 1 and MODEL_EXPECTS_CHANNELS > 1 :
        # st.warning dipanggil di UI utama jika diperlukan
        audio_data = np.tile(audio_data[:, np.newaxis], (1, MODEL_EXPECTS_CHANNELS))

    original_peak = np.max(np.abs(audio_data))
    if original_peak > 0:
        audio_data = audio_data / original_peak 
    else:
        original_peak = 1.0 
    
    if audio_data.ndim == 1: 
        input_tensor_np = audio_data[:, np.newaxis] 
    else: 
        input_tensor_np = audio_data
        
    input_tensor_np = input_tensor_np[np.newaxis, ...] 
    input_tensor = torch.from_numpy(input_tensor_np).float().to(device)

    # PERIKSA DIMENSI: Jika model Anda mengharapkan (B, C, T)
    # input_tensor = input_tensor.permute(0, 2, 1) 

    try:
        with torch.no_grad(): 
            denoised_output_tensor = model_loaded(input_tensor)
    except Exception as e:
        st.error(f"Error saat prediksi model PyTorch: {e}") # Tetap di sini untuk konteks
        st.error("Periksa bentuk input tensor dan arsitektur model Anda.")
        st.code(f"Bentuk input tensor yang diberikan ke model: {input_tensor.shape}")
        return audio_data_input, sr_input

    denoised_audio_np = denoised_output_tensor.cpu().numpy()
    denoised_audio = denoised_audio_np.squeeze()

    if original_peak > 0:
         denoised_audio = denoised_audio * original_peak

    return denoised_audio, current_sr

# --- Bagian Utama UI Streamlit ---
st.title("🎙️ Aplikasi Audio Denoising (Model PyTorch)")
st.markdown("Unggah file audio (.wav, .mp3, .flac) untuk mengurangi noise.")

# Tampilkan status impor model di sidebar SETELAH st.set_page_config
if GRUUnet2_model_class is not None:
    st.sidebar.success("Definisi model (gruunet2.py) berhasil diimpor.")
else:
    st.sidebar.error(
        "Gagal mengimpor definisi model dari 'gruunet2.py'. "
        "Pastikan file tersebut ada di direktori 'audio_denoising-master' "
        "dan berisi kelas model PyTorch yang benar (misalnya GRUUnet2_model)."
    )
    # Anda bisa menambahkan detail error impor di sini jika mau
    # try: from gruunet2 import GRUUnet2_model
    # except Exception as import_err: st.sidebar.expander("Detail Error Impor").error(import_err)


# Muat model
model_data = load_denoising_model_pytorch(MODEL_PATH, GRUUnet2_model_class)
model, current_device = (model_data if model_data is not None else (None, None))

# Tampilkan status pemuatan model di UI utama
if model is not None:
    st.success(f"Model PyTorch '{MODEL_PATH}' berhasil dimuat ke {current_device}.")
else:
    st.error(f"Gagal memuat model dari '{MODEL_PATH}'. Periksa pesan error di atas atau di sidebar.")
    if GRUUnet2_model_class is not None and not os.path.exists(MODEL_PATH):
        st.info(f"File model tidak ditemukan di: {os.path.abspath(MODEL_PATH)}")
        saves_dir = "saves"
        if os.path.exists(saves_dir):
            st.info(f"Isi direktori '{saves_dir}': {os.listdir(saves_dir)}")
            gruu_dir = os.path.join(saves_dir, "GRUUNet2-dari_tult")
            if os.path.exists(gruu_dir):
                st.info(f"Isi direktori '{gruu_dir}': {os.listdir(gruu_dir)}")


uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    st.subheader("Audio Asli")
    try:
        audio_original, sr_original = librosa.load(file_bytes, sr=None, mono=False)
        if audio_original.ndim > 1:
            st.audio(audio_original.T, sample_rate=sr_original)
        else:
            st.audio(audio_original, sample_rate=sr_original)
    except Exception as e:
        st.error(f"Tidak dapat memutar audio asli: {e}")
        st.audio(file_bytes)

    if model is not None:
        if st.button("Hilangkan Noise", type="primary"):
            # Peringatan jika model mengharapkan channel berbeda dari input
            # (Ini perlu data audio untuk diperiksa, jadi mungkin lebih baik di dalam process_audio_pytorch)
            # if audio_data_to_process.ndim == 1 and MODEL_EXPECTS_CHANNELS > 1:
            #    st.warning(f"Model mengharapkan {MODEL_EXPECTS_CHANNELS} channel, tapi input mono. Menduplikasi channel mono.")
            
            with st.spinner(f"Sedang memproses audio menggunakan {current_device}... Ini mungkin memakan waktu."):
                try:
                    file_bytes.seek(0)
                    audio_data_to_process, sr_to_process = librosa.load(file_bytes, sr=None, mono=False)
                    
                    if audio_data_to_process.ndim > 1:
                        audio_data_to_process = audio_data_to_process.T

                    denoised_audio, denoised_sr = process_audio_pytorch(audio_data_to_process, sr_to_process, model, current_device)
                    
                    st.subheader("Audio Setelah Denoising")
                    st.audio(denoised_audio, sample_rate=denoised_sr)

                    buffer_download = BytesIO()
                    sf.write(buffer_download, denoised_audio, denoised_sr, format='WAV', subtype='PCM_16')
                    buffer_download.seek(0)
                    
                    st.download_button(
                        label="Unduh Audio Denoised (WAV)",
                        data=buffer_download,
                        file_name=f"denoised_{uploaded_file.name.split('.')[0]}.wav",
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat proses denoising: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    else:
        st.warning("Model tidak berhasil dimuat. Fitur denoising tidak tersedia.")
else:
    st.info("Silakan unggah file audio untuk memulai.")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    f"Aplikasi ini menggunakan model deep learning PyTorch (GRUUnet2) untuk "
    f"mereduksi noise pada file audio. "
    f"Model diharapkan berada di: '{MODEL_PATH}'. "
    f"Pastikan file definisi model (misalnya 'gruunet2.py') ada dan benar."
)
st.sidebar.markdown("Dibuat dengan [Streamlit](https://streamlit.io) dan [PyTorch](https://pytorch.org/).")

# Informasi tambahan untuk debugging path
st.sidebar.subheader("Informasi Debug Path")
st.sidebar.text(f"Direktori kerja saat ini: {os.getcwd()}")
st.sidebar.text(f"Path model yang dicari: {os.path.abspath(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    st.sidebar.success("File model ditemukan.")
else:
    st.sidebar.error("File model TIDAK ditemukan pada path di atas.")