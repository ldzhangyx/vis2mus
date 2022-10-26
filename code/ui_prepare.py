from PyQt5.Qt import *
from model import DisentangleVAE
import torch
from quickdraw.names import *
import muspy
import pretty_midi
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from torch.distributions import Normal
from amc_dl.torch_plus.train_utils import get_zs_from_dists
from PIL import Image
from PIL import ImageEnhance
from pr_visualization import *

import pygame
# mixer config
freq = 44100  # audio CD quality
bitsize = -16  # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024  # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(1.0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DisentangleVAE.init_model()
model.load_model("./result/models/disvae-nozoth_final.pt")
model.eval()


def listdir(path, list_name):
    for file in os.listdir(path):
        if file[-4:] == '.jpg' or file[-4:] == '.png':
            file_path = os.path.join(path, file)
            list_name.append(file_path)


styles = []
listdir('./resources/style_img', styles)

style_names = []
for style in styles:
    style_names.append(style.split('/')[-1].split('\\')[-1][:-4])

names = []
for category in QUICK_DRAWING_NAMES:
    if ' ' in category:
        names.append(category.replace(" ", '-'))
    else:
        names.append(category)
names.remove('circle')


def from_bin(dur):
    return 1 + dur[4] * 1 + dur[3] * 2 + dur[2] * 4 + dur[1] * 8 + dur[0] * 16


def x_to_pr(x):
    if not (type(x) is np.ndarray):
        x = x.numpy()

    pr = np.zeros((1, 32, 128))
    for t in range(32):
        for i in range(x.shape[2]):
            if x[0, t, i, 0] > 128:
                break
            if  x[0, t, i, 0] < 128:
                pitch = x[0, t, i, 0]
                duration = from_bin(x[0, t, i, 1:])
                pr[0, t, pitch] = duration
    return pr


def lx_to_lpr(lx):
    x1 = lx[:, 0, :, :, :]
    x2 = lx[:, 1, :, :, :]
    x3 = lx[:, 2, :, :, :]
    x4 = lx[:, 3, :, :, :]
    pr_mat1 = x_to_pr(x1)
    pr_mat2 = x_to_pr(x2)
    pr_mat3 = x_to_pr(x3)
    pr_mat4 = x_to_pr(x4)
    return torch.tensor(np.stack((pr_mat1, pr_mat2, pr_mat3, pr_mat4), 1)).to(torch.float32)


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
                if x[0, t, i, 0] < 128:
                    pitch = x[0, t, i, 0]
                    duration = from_bin(x[0, t, i, 1:])
                    note = np.array([t + 32 * n_bar, pitch, duration, 100])
                    notes.append(note)
    return np.array(notes)


def longx_to_midi(longx, name, qpm=120.0):
    notes = longx_to_notes(longx)

    if notes.shape[0] == 0:
        notes = np.array([[0, 0, 0, 0]])
    print(notes.dtype)

    muspy_object = muspy.from_note_representation(notes, resolution=4)
    muspy_object.adjust_resolution(24)
    muspy_object.tempos = [muspy.Tempo(0, qpm)]

    muspy_object.time_signatures = [muspy.TimeSignature(0, 4, 4)]
    path = name+'.midi'
    muspy_object.tracks[0].program = 0
    muspy.write_midi(path, muspy_object)


def mel_visualization(midi_file, path='./resources/content_pr.png'):

    music = muspy.read_midi(midi_file)
    multi_track = to_pypr(music)
    plot_multitrack(multi_track, None, preset='plain', label='off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')


def longx_visualization(longx, name):
    notes = longx_to_notes(longx)
    if notes.size > 0:
        notes[:, 3] = 80
    muspy_object = muspy.from_note_representation(notes, resolution=4)
    muspy_object.adjust_resolution(24)
    multitrack = to_pypr(muspy_object)
    plot_multitrack(multitrack, None, preset='plain', label='off')
    plt.savefig(name+'.png', bbox_inches='tight',pad_inches=0.0)
    plt.close('all')


def shift_chord(chord, shift=0):
    c = chord.clone().detach()
    c[:, :, 0:12] = torch.roll(c[:, :, 0:12], shift, -1)
    c[:, :, 12:24] = torch.roll(c[:, :, 12:24], shift, -1)
    c[:, :, 24:36] = torch.roll(c[:, :, 24:36], shift, -1)
    return c


def merge_x_to_notes(long_mel1, shift, longx2):
    mel_notes = muspy.to_note_representation(muspy.read_midi('./resources/content.midi'))
    mel_notes[:, 0] = mel_notes[:, 0] / 6
    mel_notes[:, 2] = mel_notes[:, 2] / 6
    mel_notes[:, 3] = 100
    acc_notes = longx_to_notes(longx2)
    mel_notes[:, 1] = mel_notes[:, 1] + shift
    if acc_notes.size > 0:
        acc_notes[:, 3] = 80
        return np.concatenate((mel_notes, acc_notes))
    else:
        return mel_notes


def merge_x_to_midi(long_mel1, shift, longx2, name, qpm=120.0):
    notes = merge_x_to_notes(long_mel1, shift, longx2)
    muspy_object = muspy.from_note_representation(notes, resolution=4)
    muspy_object.adjust_resolution(24)
    muspy_object.tempos = [muspy.Tempo(0, qpm)]
    path = name+'.midi'

    muspy_object.tracks[0].program = 0
    muspy.write_midi(path, muspy_object)


def merge_x_visualization(long_mel1, shift, longx2, name):
    notes = merge_x_to_notes(long_mel1, shift, longx2)
    muspy_object = muspy.from_note_representation(notes, resolution=4)
    muspy_object.adjust_resolution(24)
    multi_track = to_pypr(muspy_object)
    plot_multitrack(multi_track, None, preset='plain', label='off')
    plt.savefig(name+'.png', bbox_inches='tight',pad_inches=0.0)
    plt.close('all')


def long2short(lc, lp):
    c1 = lc[:, 0:8, :]
    c2 = lc[:, 8:16, :]
    c3 = lc[:, 16:24, :]
    c4 = lc[:, 24:32, :]

    pr_mat1 = lp[:, 0, :, :]
    pr_mat2 = lp[:, 1, :, :]
    pr_mat3 = lp[:, 2, :, :]
    pr_mat4 = lp[:, 3, :, :]
    return c1, c2, c3, c4, pr_mat1, pr_mat2, pr_mat3, pr_mat4


def contrast(origin, percent):
    average = origin.mean()
    out = average + (origin - average) * (1 + percent / 100.0)
    return np.maximum(out, 0)


def modify(n_bar, pr_mat, c, val1, val2, val3):
    maps = model.rhy_encoder.cnn(pr_mat.unsqueeze(1)).detach().numpy()
    dmaps = np.zeros(maps.shape)
    for i in range(0, 10):
        res = np.maximum(0, maps[0, i] + val1/100.0 * maps[0, i].max() / 2)
        if val2 != 0:
            res = contrast(res, val2)
        dmaps[0, i] = res

    with torch.no_grad():
        pr = torch.tensor(dmaps).view(1, 8, -1).to(torch.float32)
        pr = model.rhy_encoder.fc2(model.rhy_encoder.fc1(pr))  # (bs, 8, emb_size)
        pr = model.rhy_encoder.gru(pr)[-1]
        pr = pr.transpose_(0, 1).contiguous()
        pr = pr.view(pr.size(0), -1)
        mu = model.rhy_encoder.linear_mu(pr)
        var = model.rhy_encoder.linear_var(pr).exp_()
        dist = Normal(mu, var)

        dist_chd = model.chd_encoder(c)
        dist_rhy = dist
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], False)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1)
        pitch_outs, dur_outs = model.decoder(dec_z, True, None, None, 0., 0.)
        est_x, _, _ = model.decoder.output_to_numpy(pitch_outs, dur_outs)

    return est_x


def long_modify(lx, lc, lp, val1, val2, val3):

    c1, c2, c3, c4, pr_mat1, pr_mat2, pr_mat3, pr_mat4 = long2short(lc, lp)

    out_x1 = modify(1, pr_mat1, c1, val1, val2, val3)
    out_x2 = modify(2, pr_mat2, c2, val1, val2, val3)
    out_x3 = modify(3, pr_mat3, c3, val1, val2, val3)
    out_x4 = modify(4, pr_mat4, c4, val1, val2, val3)
    out_x = np.stack((out_x1, out_x2, out_x3, out_x4), 1)

    return out_x


def numpy_to_midi(sample_roll, output='sample.mid', qpm=120.):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    t = 0

    for i in sample_roll:
        if 'torch' in str(type(i)):
            pitch = int(i.max(0)[1])
        else:
            pitch = int(np.argmax(i))
        if pitch < 128:
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
            t += 1 / 8
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pretty_midi.Note(
                    velocity=100, pitch=int(p), start=0, end=t)
            note = pretty_midi.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + 1 / 8)
            piano.notes.append(note)
            t += 1 / 8
        elif pitch == 129:
            t += 1 / 8
    music.instruments.append(piano)
    muspy_obj = muspy.from_pretty_midi(music)
    muspy_obj.tempos = [muspy.Tempo(0, qpm)]
    muspy.write_midi(output, muspy_obj)


class MusicPlayer(QWidget):
    def __init__(self):

        super().__init__()
        self.setup()
        self.playing = False
        self.Is_Null = False

    def set_song(self, song):
        self.song = song

    def set_qpm(self, qpm):
        self.qpm = float(qpm)

    def play(self):
        if self.playing:
            self.animation.stop()

        self.parent().setFixedSize(self.parent().size())

        self.scanline.raise_()
        self.scanline.show()
        self.scanline.setFixedSize(1, self.label.height())
        self.animation.setStartValue(self.label.pos())
        self.animation.setEndValue(QPoint(self.label.x() + self.label.width(), self.label.y()))
        self.animation.setDuration(int(32 * 60 / self.qpm * 1000))

        pygame.mixer.music.load(self.song)
        if self.Is_Null == False:
            pygame.mixer.music.play()
        self.animation.start()
        self.playing = True

    def stop(self):
        if self.playing:
            pygame.mixer.music.stop()
            self.animation.stop()
            self.parent().setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
            self.parent().setMinimumSize(1018, 558)
            self.scanline.hide()
        self.playing = False

    def reset(self):
        self.animation.stop()
        self.scanline.move(self.label.pos())
        self.parent().setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        self.parent().setMinimumSize(1018, 558)
        self.scanline.hide()

    def setup(self):

        self.label = QLabel()
        self.label.setObjectName("pr")
        self.label.setScaledContents(True)

        play_btn = QPushButton('Play')
        play_btn.setObjectName('play')
        play_btn.clicked.connect(self.play)
        stop_btn = QPushButton('Stop')
        stop_btn.clicked.connect(self.stop)

        self.scanline = QLabel(self)
        self.scanline.setPixmap(QPixmap('resources/scanline.png'))
        self.scanline.setScaledContents(True)
        self.scanline.hide()

        self.animation = QPropertyAnimation(self.scanline, b'pos', self)

        def judge_playing(ns):
            if ns == 0:
                self.playing = False

        self.animation.stateChanged.connect(judge_playing)

        hl = QHBoxLayout()
        hl.addWidget(play_btn)
        hl.addWidget(stop_btn)

        vl = QVBoxLayout()
        vl.addWidget(self.label, alignment=Qt.AlignCenter)
        vl.addLayout(hl)

        self.setLayout(vl)


class MySliderLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()

    def set_argu(self, text, a, b):
        self.text = text
        self.a = a
        self.b = b
        self.setup()

    def reset(self):
        pass

    def val(self):
        return self.slider.value()

    def setup(self):
        label = QLabel()
        label.setText(self.text)

        vl = QVBoxLayout()
        value_label = QLabel("0%")
        value_label.setAlignment(Qt.AlignCenter)
        warn_label = QLabel()
        warn_label.setText("️ ")

        warn_label.setAlignment(Qt.AlignCenter)
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMaximum(100)
        self.slider.setMinimum(-100)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setMinimumWidth(200)
        self.slider.setTickPosition(QSlider.TicksBelow)

        def display_value(val):
            if val > self.b or val < self.a:
                value_label.setStyleSheet("color:red")
                warn_label.setStyleSheet("color:red")
                warn_label.setText("️Out of the recommended range!")
            else:
                value_label.setStyleSheet("color:black;")
                warn_label.setText("️ ")
            value_label.setText(str(val) + '%')

        self.slider.valueChanged.connect(display_value)

        vl.addWidget(value_label)
        vl.addWidget(self.slider)
        vl.addWidget(warn_label)

        self.reset_btn = QPushButton('Reset')

        self.addWidget(label, 0, alignment = Qt.AlignLeft)
        self.addLayout(vl, 10)
        self.addWidget(self.reset_btn, 0)

