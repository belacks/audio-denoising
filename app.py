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


def save_model(
    name, 
    model, 
    optimizer=None, 
    scheduler=None, 
    arch=None, 
    last_epoch=None,
    loss_record=None, 
    loss_metric=None,
    total_training_iters=None,
    last_target_name=None, 
    last_batch_size=None,
    last_dataset_name=None,
    tag_uuid=True,
    or_tag_date=True,
    allow_overwrite=False,
    prefix='saves',
):
    import inspect
    if tag_uuid:
        import uuid
        name=name+'-'+uuid.uuid4().hex[:6]
    elif or_tag_date:
        import datetime
        suffix=datetime.datetime.now().strftime("%y%m%d")
        name=name+'-'+suffix
    if os.path.exists(os.path.join(prefix,name)) and not allow_overwrite:
        raise FileExistsError("File/dir already exists")
    elif not allow_overwrite:
        os.makedirs(os.path.join(prefix,name))
    #with open(os.path.join('saves',name,"source_code.py")) as f:
    #    f.write(inspect.getsource(instance))
    checkpoint = {
        'last_epoch': last_epoch,  # current training epoch
        'loss_record': loss_record if loss_record is not None else dict(),  
        'loss_metric': loss_metric, 
        'last_target_name': last_target_name,  
        'total_training_iters': total_training_iters,  # total_training_iters
        'arch': arch, 
        'last_batch_size': last_batch_size, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': model.get_config(),  # a dict of hyperparameters
        'arch': arch if arch is not None else type(model).__name__,      # e.g., obtained via subprocess from git
        # Optionally add any additional info (loss, metrics, etc.)
    }
    
    torch.save(checkpoint, os.path.join(prefix,name,'checkpoint.pth'))
DEVICE=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
class TrainingContext:
    def __init__(self, cls, *args, **kwargs):
        self.inner = cls(*args, **kwargs).to(DEVICE)
        self.name = cls.__name__
        self.optim = torch.optim.AdamW(self.inner.parameters())
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.9)
        self.num_parameters = sum(map(torch.numel,self.inner.parameters()))
        self.train_loss_record = dict()
        self.test_loss_record = dict()
        self.results = list()
        self.total_iters = 0
        self.running_loss = 0
        self.best_eval_loss = 999
        self.stopped = False
        self.batch_size=64
        self.train_loss_metric='L1'
        self.eval_loss_metric='L1'
        self.last_target_name='clamped mel-spectrogram'
        self.last_dataset_name='miscellaneous'
        self.training=True
    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)
    def __getattr__(self, name):
        return self.inner.__getattribute__(name)
    def save(self, prefix='saves'):
        save_model(
            self.name,
            self.inner,
            optimizer=self.optim,
            scheduler=self.sched,
            loss_record={
                'train':self.train_loss_record,
                'test':self.test_loss_record,
            },
            total_training_iters=self.total_iters,
            last_batch_size=self.batch_size,
            loss_metric={
                'train':'MSE',
                'test':'MAE',
            },
            last_target_name=self.last_target_name,
            last_dataset_name=self.last_dataset_name
        )
    @classmethod
    def load(cls, name, class_, prefix='saves', training=False):
        checkpoint=torch.load(os.path.join(prefix,name,'checkpoint.pth'),map_location=DEVICE)
        self=cls(class_,**checkpoint['config'])
        self.inner.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.sched.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_iters=checkpoint['total_training_iters']
        self.batch_size=checkpoint['last_batch_size']
        self.train_loss_record=checkpoint['loss_record']['train']
        self.test_loss_record=checkpoint['loss_record']['test']
        self.best_eval_loss=min(self.test_loss_record.values()) if len(self.test_loss_record.values())>0 else None
        self.training=training
        return self
    @classmethod
    def _load(cls, class_, path, training=False):
        checkpoint=torch.load(path,map_location=DEVICE)
        self=cls(class_,**checkpoint['config'])
        self.inner.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.sched.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_iters=checkpoint['total_training_iters']
        self.batch_size=checkpoint['last_batch_size']
        self.train_loss_record=checkpoint['loss_record']['train']
        self.test_loss_record=checkpoint['loss_record']['test']
        self.best_eval_loss=min(self.test_loss_record.values()) if len(self.test_loss_record.values())>0 else None
        self.training=training
        return self.inner

# --- Fungsi untuk Memuat Model PyTorch ---
@st.cache_resource # Cache model agar tidak di-load ulang setiap interaksi
def load_denoising_model_pytorch(model_path, ModelClass):
    return TrainingContext._load(ModelClass,model_path)

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
model = load_denoising_model_pytorch(MODEL_PATH, GRUUnet2_model_class)
#model, current_device = (model_data if model_data is not None else (None, None))
current_device=DEVICE
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