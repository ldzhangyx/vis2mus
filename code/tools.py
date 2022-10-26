from model import DisentangleVAE
from ptvae import RnnEncoder, TextureEncoder, PtvaeEncoder, PtvaeDecoder, \
    RnnDecoder
from dataset_loaders import MusicDataLoaders, TrainingVAE
from dataset import SEED
from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, MinExponentialLR, \
    TeacherForcingScheduler, ConstantScheduler
from amc_dl.torch_plus.train_utils import kl_anealing
import torch
from torch import optim
import numpy as np
import muspy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.distributions import Normal
from amc_dl.torch_plus.train_utils import get_zs_from_dists
from PIL import Image, ImageEnhance
from converter import target_to_3dtarget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
readme_fn = './train.py'

n_epoch = 6
clip = 1
parallel = False
parallel = parallel if torch.cuda.is_available() and \
                       torch.cuda.device_count() > 1 else False

name = 'disvae-nozoth'
batch_size = 1  # 做inference时一次只需要一个片段

model = DisentangleVAE.init_model()
model.load_model("./result/models/disvae-nozoth_final.pt")
model.eval()


def batch_to_inputs(batch):
    _, _, pr_mat, x, c, dt_x = batch
    pr_mat = pr_mat.to(device).float()
    x = x.to(device).long()
    c = c.to(device).float()
    dt_x = dt_x.to(device).float()
    return x, c, pr_mat, dt_x


def from_bin(dur):
    return 1 + dur[4] * 1 + dur[3] * 2 + dur[2] * 4 + dur[1] * 8 + dur[0] * 16


def x_to_notes(x):
    if not (type(x) is np.ndarray):
        x = x.numpy()

    notes = []
    for t in range(32):
        for i in range(x.shape[2]):
            if x[0, t, i, 0] > 128:
                break
            if (x[0, t, i, 0] < 128):
                pitch = x[0, t, i, 0]
                duration = from_bin(x[0, t, i, 1:])
                note = np.array([t, pitch, duration, 100])
                notes.append(note)
    return np.array(notes)


def x_to_midi(x, name):
    path = './exp/' + name
    notes = x_to_notes(x)
    muspy_object = muspy.from_note_representation(notes, resolution=4)
    muspy.write_midi(path, muspy_object)


def longx_to_notes(longx):
    if not (type(longx) is np.ndarray):
        longx = longx.numpy()
    notes = []
    for n_bar in range(longx.shape[1]):
        x = longx[:, n_bar, :, :, :]
        for t in range(32):
            for i in range(x.shape[2]):
                if x[0, t, i, 0] > 128:
                    break
                if (x[0, t, i, 0] < 128):
                    pitch = x[0, t, i, 0]
                    duration = from_bin(x[0, t, i, 1:])
                    note = np.array([t + 32 * n_bar, pitch, duration, 100])
                    notes.append(note)
    return np.array(notes)


def longx_to_midi(longx, name):
    path = './exp/' + name
    notes = longx_to_notes(longx)
    muspy_object = muspy.from_note_representation(notes, resolution=4)
    muspy.write_midi(path, muspy_object)

