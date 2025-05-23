import av
import sounddevice as sd

import itertools

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn

import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram, AmplitudeToDB, InverseSpectrogram, InverseMelScale, GriffinLim, MuLawEncoding, MuLawDecoding
import matplotlib.pyplot as plt
from contextlib import contextmanager
from joblib import Memory
import random
import os

def get_canonical_filename(filename):
    return os.path.realpath(os.path.normcase(os.path.abspath(filename)))
    
AUDIO_CACHE=dict()
#print("AUDIO_CACHE id:",id(AUDIO_CACHE))
SR=48000
# n_fft=400
# win_length=100
# hop_length=50

# n_fft = 480
# win_length = int(0.01 * SR) 
# hop_length = int(0.02/0.025 * win_length) 
# print("n_fft:",n_fft)
# print("win_length:",win_length)
# print("hop_length:",hop_length)
# #n_fft = 1200
# #win_length = int(0.025 * SR)  # 25 ms window => 1200 samples
# #hop_length = int(0.010 * SR)  # 10 ms hop => 480 samples

# T=Spectrogram(power=None, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
# I=InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)

# MU=MuLawEncoding()
# IMU=MuLawDecoding()

R1=torchaudio.transforms.Resample(44100,SR)
R2=torchaudio.transforms.Resample(SR,44100)

@contextmanager
def figsize_as(width, height):
    original_figsize = plt.rcParams['figure.figsize']
    try:
        plt.rcParams['figure.figsize'] = [width, height]
        yield
    finally:
        plt.rcParams['figure.figsize'] = original_figsize

memory = Memory('cache', verbose=0)
@memory.cache
def get_num_samples(filename):
    container = av.open(filename)
    n_samples=0
    for frame in container.decode(audio=0):
        n_samples+=frame.samples
    return n_samples


def unwrap_complex(z):
    #print("unwrap_complex",z.shape,z.device)
    return torch.stack([z.real,z.imag]).transpose(0,1)
    
def wrap_complex(x,device=torch.tensor([]).device):
    #print("wrap_complex",x.shape,x.device)
    xt=x.transpose(0,1)
    tmp=torch.zeros(*xt.shape[1:],dtype=torch.complex64,device=device)
    tmp.real=xt[0]
    tmp.imag=xt[1]
    return tmp

def clamp(x):
    #print("clamp",x.shape,x.device)
    #y=torch.zeros_like(x, dtype=x.dtype)
    #pos=(-1<x)&(x<1)
    #y[pos]=x[pos]
    #y[~pos]=(x[~pos].abs().log()+1)*x[~pos].sign()
    return (x.abs()+1).log()*x.sign()
def unclamp(y):
    #print("unclamp",y.shape,y.device)
    #x=torch.zeros_like(y, dtype=y.dtype)
    #pos=(-1<y)&(y<1)
    #x[pos]=y[pos]
    #x[~pos]=torch.exp(y[~pos].abs()-1)*y[~pos].sign()
    return y.sign()*(y.abs().exp()-1)


def get_random_audio_buffer(filename, buffer_size, use_torch=True):
    global AUDIO_CACHE
    __filename=get_canonical_filename(filename)
    #print("id(AUDIO_CACHE):",id(AUDIO_CACHE))
    #print("AUDIO_CACHE.keys():",AUDIO_CACHE.keys())
    #print(__filename in AUDIO_CACHE)
    cache_entry=AUDIO_CACHE[__filename]
    n_samples = (cache_entry['num_samples'])
    sample_rate = (cache_entry['sample_rate'])
    ptr = np.random.randint(0,n_samples-buffer_size)
    sample = (cache_entry['samples'])[...,ptr:ptr+buffer_size]
    if 'int8' in str(sample.dtype):
        sample=sample.astype('float32')/128.0
    if 'int16' in str(sample.dtype):
        sample=sample.astype('float32')/32768.0
    if 'int32' in str(sample.dtype):
        sample=sample.astype('float32')/2147483648.0
    if 'int64' in str(sample.dtype):
        sample=sample.astype('float32')/9223372036854775808.0
    if use_torch:
        return torch.tensor(sample), sample_rate
    return (sample), sample_rate

def collect_random_audio_until_meets_buffer(filenames, buffer_size, use_torch=True):
    global AUDIO_CACHE
    filename=random.choice(filenames)
    #print("processing",filename)
    __filename=get_canonical_filename(filename)
    cache_entry=AUDIO_CACHE[__filename]
    n_samples = (cache_entry['num_samples'])
    sample_rate = (cache_entry['sample_rate'])
    samples = cache_entry['samples']
    if 'int8' in str(samples.dtype):
        samples=samples.astype('float32')/128.0
    if 'int16' in str(samples.dtype):
        samples=samples.astype('float32')/32768.0
    if 'int32' in str(samples.dtype):
        samples=samples.astype('float32')/2147483648.0
    if 'int64' in str(samples.dtype):
        samples=samples.astype('float32')/9223372036854775808.0
    while n_samples<buffer_size:
        #filename=random.choice(filenames)
        __filename=get_canonical_filename(filename)
        cache_entry=AUDIO_CACHE[__filename]
        if (cache_entry['sample_rate']!=sample_rate):
            #print("skipping",filename,'... its sample rate...',cache_entry['sample_rate'],'is different from',sample_rate)
            continue
        #print("appending",filename)
        n_samples += (cache_entry['num_samples'])
        _samples=cache_entry['samples']
        if 'int8' in str(_samples.dtype):
            _samples=_samples.astype('float32')/128.0
        if 'int16' in str(_samples.dtype):
            _samples=_samples.astype('float32')/32768.0
        if 'int32' in str(_samples.dtype):
            _samples=_samples.astype('float32')/2147483648.0
        if 'int64' in str(_samples.dtype):
            _samples=_samples.astype('float32')/9223372036854775808.0
        samples = np.concatenate([samples, _samples],-1)
    if n_samples>buffer_size:
        #print()
        ptr = np.random.randint(0,n_samples-buffer_size)
        samples = samples[...,ptr:ptr+buffer_size]
            
    if use_torch:
        return torch.tensor(samples), sample_rate
    return (samples), sample_rate
    
def stream_random_audio_buffer(filenames, buffer_size, use_torch=True, max_samples=1):
    n_samples=0
    while n_samples<max_samples:
        samples, sample_rate = collect_random_audio_until_meets_buffer(filenames, buffer_size, use_torch=use_torch)
        n_samples+=samples.shape[-1]
        yield samples, sample_rate

def plot(tensor,*args,**kwargs):
    return plt.plot(tensor.view(-1).detach().cpu().numpy(),*args,**kwargs)
def imshow(tensor,*args,**kwargs):
    return plt.imshow(tensor.detach().cpu().numpy(),*args,**kwargs)

def read_audio(file_path, max_samples=2<<63, use_tensor=True):
    container = av.open(file_path)
    samples=[]
    n_samples=0
    for frame in container.decode(audio=0):
        _samples=frame.to_ndarray()
        n_samples+=_samples.shape[-1]
        samples.append(_samples)
        if n_samples>max_samples:
            break
    samples = np.concatenate(samples,axis=-1)[..., :max_samples]
    if 'int8' in str(samples.dtype):
        samples=samples.astype('float32')/128.0
    if 'int16' in str(samples.dtype):
        samples=samples.astype('float32')/32768.0
    if 'int32' in str(samples.dtype):
        samples=samples.astype('float32')/2147483648.0
    if 'int64' in str(samples.dtype):
        samples=samples.astype('float32')/9223372036854775808.0
    if use_tensor:
        samples=torch.tensor(samples)
    return samples, frame.sample_rate

def play_audio(ndarray, sample_rate, blocking=True):
    """
    Plays a stereo or mono audio signal represented as a NumPy ndarray.

    Args:
        ndarray (numpy.ndarray): The audio data with shape `(channels, n_samples)`.
        sample_rate (int): The sample rate of the audio.
    """
    if isinstance(ndarray, torch.Tensor):
        ndarray=ndarray.detach().cpu().numpy()
    if len(ndarray.shape) == 1:  # Mono audio, reshape to (1, n_samples)
        ndarray = ndarray[np.newaxis, :]
    if ndarray.shape[0] > 2:
        raise ValueError("Audio playback only supports mono (1 channel) or stereo (2 channels).")
    
    # Transpose to shape (n_samples, channels) as sounddevice expects this
    sd.play(ndarray.T, samplerate=sample_rate)
    if blocking:
        sd.wait()  # Wait until the audio is finished playing

def buffer_stream(audio_stream_generator, buffer_size, use_torch=True, limit_samples=10**20, skip_samples=0):
    "assumes the sample rate is uniform across the audio stream"
    buffer=[]
    cur_size=0
    total_samples=0
    skipped_samples=0
    __sample_rate=None
    for samples, sample_rate in audio_stream_generator:
        if not __sample_rate:
            __sample_rate=sample_rate
        assert __sample_rate==sample_rate, "sample rate must be consistent"
        n_samples=samples.shape[-1]
        
        if skipped_samples<skip_samples:
            skipped_samples+=n_samples
            continue
            
        total_samples+=n_samples
        cur_size+=n_samples
            
        buffer.append(samples)
        if total_samples>limit_samples:
            break
        if cur_size>=buffer_size:
            #print('buffer_stream: buffer is now sized',cur_size,'exceeding',buffer_size)
            if use_torch:
                cat= torch.cat(buffer,-1)
            else:
                cat= np.concatenate(buffer,-1)
            #print("buffer_stream: cat:",cat.shape)
            for i in range(cat.shape[-1]//buffer_size):
                #print(f'buffer_stream: #{i}',"yielding from",buffer_size*i,'to',(1+i)*buffer_size)
                yield cat[:,buffer_size*i:(1+i)*buffer_size],__sample_rate
            #print("buffer_stream: resizing buffer and cur_size")
            buffer=[cat[...,buffer_size:]]
            cur_size=cur_size-buffer_size
            #print('buffer_stream: len(buffer):',len(buffer))
            #print('buffer_stream: buffer[-1]:',buffer[-1].shape)
            #print('buffer_stream: cur_size:',cur_size)
    if cur_size>0:
        #print("buffer_stream: residue detected")
        if use_torch:
            cat= torch.cat(buffer,-1)
        else:
            cat= np.concatenate(buffer,-1)
        #print("buffer_stream: cat:",cat.shape)
        for i in range(cat.shape[-1]//buffer_size):
            #print(f'buffer_stream: #{i}',"yielding from",buffer_size*i,'to',(1+i)*buffer_size)
            yield cat[:,buffer_size*i:(1+i)*buffer_size],__sample_rate



def __stream_audio(file_path, use_torch=True, discard_first=False, use_cache=True):
    global AUDIO_CACHE
    #print("__stream_audio: id(AUDIO_CACHE):",id(AUDIO_CACHE))
    __file_path=get_canonical_filename(file_path)
    if use_cache and (AUDIO_CACHE.get(__file_path) is not None):
        #print("__stream_audio: using available cache...")
        entry=AUDIO_CACHE.get(__file_path)
        sample_rate=entry["sample_rate"]
        #print("__stream_audio: sample_rate:", sample_rate)
        is_first=True
        for i in range(0,entry["num_samples"]-entry["frame_size"],entry["frame_size"]):
            #print(f"__stream_audio: #{i}")
            if is_first and discard_first:
                #print("__stream_audio: discarding first entry")
                is_first=False
                continue
            samples = entry["samples"][...,i:i+entry["frame_size"]]
            if 'int8' in str(samples.dtype):
                samples=samples.astype('float32')/128.0
            if 'int16' in str(samples.dtype):
                samples=samples.astype('float32')/32768.0
            if 'int32' in str(samples.dtype):
                samples=samples.astype('float32')/2147483648.0
            if 'int64' in str(samples.dtype):
                samples=samples.astype('float32')/9223372036854775808.0
            #print("__stream_audio: samples:",samples.shape)
            if use_torch:
                _samples=torch.tensor(samples.copy())
                yield _samples, sample_rate
            else:
                yield samples, sample_rate
    else:
        #print("__stream_audio: no available cache found...")
        to_save=list()
        container = av.open(__file_path)
        is_first=True
        n_samples=0
        n_frames=0
        for frame in (container.decode(audio=0)):
            _samples = frame.to_ndarray()
            #print("__stream_audio: _samples:",_samples.shape)
            n_frames+=1
            n_samples += _samples.shape[-1]
            if use_cache:
                to_save.append(_samples.copy())
            
            if is_first and discard_first:
                print("__stream_audio: discarding first entry")
                is_first=False
                continue
            if 'int8' in str(_samples.dtype):
                _samples=_samples.astype('float32')/128.0
            if 'int16' in str(_samples.dtype):
                _samples=_samples.astype('float32')/32768.0
            if 'int32' in str(_samples.dtype):
                _samples=_samples.astype('float32')/2147483648.0
            if 'int64' in str(_samples.dtype):
                _samples=_samples.astype('float32')/9223372036854775808.0
            if use_torch:
                _samples=torch.tensor(_samples)
            sample_rate=frame.sample_rate
            #print("__stream_audio: sample_rate:", sample_rate)
            yield _samples, sample_rate
        if use_cache:
            #print("__stream_audio: saving cache to", __file_path)
            __to_save = np.concatenate(to_save,-1)
            AUDIO_CACHE[__file_path]={
                "sample_rate":sample_rate,
                "frame_size":n_samples//n_frames,
                "samples":__to_save,
                "num_samples":n_samples
            }

def stream_audio(file_path, use_torch=True, discard_first=False, buffer_size=48000, limit_samples=10**20, skip_samples=0, use_cache=True):
    return buffer_stream(
        __stream_audio(file_path, use_torch=use_torch, discard_first=discard_first, use_cache=use_cache), 
        buffer_size=buffer_size, 
        limit_samples=limit_samples,
        skip_samples=skip_samples,
        use_torch=use_torch,
    )

def limit_stream(stream, max_samples):
    total_samples=0
    for samples, sample_rate in stream:
        total_samples+=samples.shape[-1]
        yield samples, sample_rate
        if total_samples>=max_samples:
            break

def __combine_samples(s1,s2):
    if isinstance(s1, torch.Tensor):
        return (s1+s2).clamp(-1,1).clone()
    return np.clip(s1+s2,-1,1)

def combine_audio(a1, a2):
    s1, sr1 = a1
    s2, sr2 = a2
    assert sr1==sr2, "sample rates must be the same"
    return __combine_samples(s1,s2), sr1

def clip_audio_to_same_size(a1, a2):
    s1, sr1 = a1
    s2, sr2 = a2
    assert sr1==sr2, "sample rates must be the same"
    if s1.shape[-1]>s2.shape[-1]:
        return (
            (
                s1[...,:s2.shape[-1]],
                sr1
            ),
            (
                s2,
                sr1
            ),
        )
    return (
            (
                s1,
                sr1
            ),
            (
                s2[...,:s1.shape[-1]],
                sr1
            ),
        )


STDS=torch.tensor([0.3922, 0.2043, 0.2245, 0.1914, 0.1832, 0.1889, 0.1823, 0.1581, 0.1304,
        0.1081, 0.0921, 0.0825, 0.0775, 0.0758, 0.0749, 0.0713, 0.0643, 0.0567,
        0.0501, 0.0443, 0.0398, 0.0376, 0.0366, 0.0371, 0.0376, 0.0372, 0.0356,
        0.0324, 0.0289, 0.0254, 0.0231, 0.0221, 0.0214, 0.0218, 0.0223, 0.0227,
        0.0227, 0.0221, 0.0209, 0.0192, 0.0173, 0.0159, 0.0150, 0.0141, 0.0130,
        0.0123, 0.0119, 0.0112, 0.0107, 0.0101, 0.0098, 0.0097, 0.0095, 0.0095,
        0.0097, 0.0096, 0.0098, 0.0099, 0.0096, 0.0094, 0.0092, 0.0090, 0.0088,
        0.0086, 0.0084, 0.0081, 0.0079, 0.0077, 0.0075, 0.0073, 0.0072, 0.0072,
        0.0070, 0.0068, 0.0067, 0.0066, 0.0067, 0.0066, 0.0065, 0.0064, 0.0065,
        0.0066, 0.0068, 0.0068, 0.0068, 0.0067, 0.0067, 0.0066, 0.0065, 0.0065,
        0.0064, 0.0063, 0.0063, 0.0063, 0.0063, 0.0063, 0.0062, 0.0062, 0.0061,
        0.0062, 0.0062, 0.0062, 0.0061, 0.0061, 0.0062, 0.0062, 0.0063, 0.0062,
        0.0062, 0.0061, 0.0060, 0.0059, 0.0060, 0.0061, 0.0060, 0.0061, 0.0061,
        0.0062, 0.0063, 0.0063, 0.0063, 0.0062, 0.0061, 0.0061, 0.0059, 0.0059,
        0.0057, 0.0056, 0.0056, 0.0055, 0.0056, 0.0056, 0.0055, 0.0055, 0.0054,
        0.0052, 0.0051, 0.0051, 0.0050, 0.0049, 0.0048, 0.0048, 0.0048, 0.0047,
        0.0047, 0.0045, 0.0044, 0.0043, 0.0043, 0.0040, 0.0029, 0.0024, 0.0021,
        0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0015, 0.0014, 0.0014, 0.0014,
        0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0011,
        0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009,
        0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009,
        0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008,
        0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008,
        0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008,
        0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008])

def normalize(x):
    if len(x.shape)==3:
        return (x)/STDS.to(x.device).view(1,-1,1)
    return (x)/STDS.to(x.device).view(1,1,-1,1)
    
def denormalize(x):
    if len(x.shape)==3:
        return (x*STDS.to(x.device).view(1,-1,1))
    return (x*STDS.to(x.device).view(1,1,-1,1))
    