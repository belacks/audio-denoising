import av
import sounddevice as sd

import itertools

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn
DEVICE=torch.device("cpu")
#torch.set_default_device(DEVICE)#('cpu:0')

import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram, AmplitudeToDB, MelScale, InverseMelScale, GriffinLim, InverseSpectrogram
from torchvision.transforms import Resize, InterpolationMode

from utils import *

import random

import pandas as pd

#from models import *

from utils import *
import time
from multiprocessing.connection import Listener
#import gruunet2
#reload(gruunet2)
from   gruunet import GRUUNet
from   gruunet2 import GRUUNet2

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
        self.last_target_name='clamped raw-spectrogram'
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


#R1=torchaudio.transforms.Resample(44100,SR)
#R2=torchaudio.transforms.Resample(SR,44100)

print("initializing model")
#model=(CombinedModel())
#model.load_state_dict(torch.load("good_small_model2.pth"))
model = TrainingContext.load("GRUUNet2-good",GRUUNet2).inner.to(DEVICE)
print("loaded model ",type(model))
#print("testing model...")
#START=time.time()
#for i in range(100):
#    W0 = np.random.rand(4800,2).astype(np.float32)
#    Wi = (torch.tensor(W0.T, device='cpu')).cuda()
#    X = clamp(normalize(unwrap_complex(T(Wi))))
#    with torch.no_grad():
#        H = model(X)
#    O = I(wrap_complex(denormalize(unclamp(H))))
#    Wo = (O.cpu()).T.numpy()
#END=time.time()
#print("time to process 1000x4800x2 samples:",END-START)
#print("averaged duration:",(END-START)/1000)
n_fft = 1024#600
n_mels = 64#22
n_stft=n_fft//2+1
win_length = n_fft
hop_length = win_length//2
# T0: waveform -> spectrogram
# I0: spectrogram -> waveform
T0=Spectrogram(power=None, n_fft=n_fft, win_length=win_length, hop_length=hop_length).to(DEVICE)
I0=InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length).to(DEVICE)
M0T=MelScale(n_mels=n_mels,n_stft=n_stft,sample_rate=SR).to(DEVICE)
M0I=InverseMelScale(n_mels=n_mels,n_stft=n_stft,sample_rate=SR).to(DEVICE)
hx=None
address = ('localhost', 6101)     # family is deduced to be 'AF_INET'
n_channels=1
old_shape=(0,0)
while True:
    try:
        with Listener(address) as listener:
            listener._listener._socket.settimeout(5)
            print("listening...")
            while True:
                with listener.accept() as conn:
                    print("got a connection!")
                    while True:
                        try:
                            X = conn.recv()
                            old_shape=X.shape
                        except:
                            print("got an error... closing connection...")
                            conn.close()
                            break
                        if isinstance(X,str) and X == 'close':
                            print("closing connection...")
                            conn.close()
                        X=torch.tensor(X,device=DEVICE).T#(2,1023)
                        #print("received X:",X.shape)
                        
                        if len(X.shape)==2:
                            n_channels=X.shape[0]
                            X=X[0].view(1,-1)#monotize
                        #print("X:",X.shape)
                        abs_spec = T0(X)
                        phase=abs_spec.angle()
                        magn = abs_spec.abs()
                        log_mel_mag = M0T(magn).log1p()
                        with torch.no_grad():
                            out,hx = model(log_mel_mag.transpose(-1,-2), hx)
                            out = nn.functional.leaky_relu(out.transpose(-1,-2),negative_slope=0)*3
                            hx = hx*0.9 
                        O = M0I((log_mel_mag-out).exp()-1)
                        O = I0(torch.polar(O, phase)).repeat(n_channels,1)#(2,1023)
                        #print("O:",O.shape)
                        O = O.T.cpu().numpy()
                        #print(old_shape,"->",O.shape)
                        conn.send(O)
    except KeyboardInterrupt as e:
        print("exitting...")
        break
    except Exception as e:
        print("got error:",e)
        print("restarting listener...")
        time.sleep(0.1)