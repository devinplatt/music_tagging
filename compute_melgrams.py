#!/usr/bin/env python
"""

Script to extract features from Magnatagatune, using CPU parallelism.
This script takes maybe ~30 minutes to run on a modern quad core processor.

For each file we check to see if it exists before extracting mel features,
so if you exit the script prematurely, it should run faster the next time
without redoing too much work.

Mel spectrograms are computed as by Keunwoo Choi:
https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/audio_processor.py

"""
import argparse
import datetime
import librosa
import math
from multiprocessing import Pool
import numpy as np
import os

parser = argparse.ArgumentParser(description='A script to extract mel features on Magnatagatune')
parser.add_argument('--audio_dir',
                    help='The directory containing Magnatagatune audio.')
parser.add_argument('--num_processes', type=int, default=4,
                    help='The number of processes to run in parallel.')
parser.add_argument('--dataset_dir',
                    help='The directory containing the output of create_train_valid_test_split.py.')
args = parser.parse_args()


def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


# eg. dirname/e/yongen-moonrise-01-moonrise-88-117.mp3.mel.npy
def audio_fname_to_mel_fname(audio_fname):
    return audio_fname.replace('.mp3', '.mp3.mel.npy')


def process_audio(audio_fname):
    mel_fname = audio_fname_to_mel_fname(audio_fname)
    if os.path.exists(mel_fname):
        return 0
    else:
        mel = compute_melgram(audio_fname)
        np.save(mel_fname, mel)
        return 1
    

def process_audios(audio_fnames):
    num_processed = 0
    for audio_fname in audio_fnames:
        num_processed += process_audio(audio_fname)
    return num_processed

st = datetime.datetime.now()
num_processes = args.num_processes
dirname = args.audio_dir
dataset_dir = args.dataset_dir
train_fname = os.path.join(dataset_dir, 'train.tsv')
valid_fname = os.path.join(dataset_dir, 'valid.tsv')
test_fname = os.path.join(dataset_dir, 'test.tsv')
train_fnames = [line.strip().split('\t')[0] for line in open(train_fname)]
valid_fnames = [line.strip().split('\t')[0] for line in open(valid_fname)]
test_fnames = [line.strip().split('\t')[0] for line in open(test_fname)]
all_fnames = train_fnames + valid_fnames + test_fnames
all_fnames = [os.path.join(dirname, key) for key in all_fnames]
p = Pool(num_processes)
num_each = int(math.ceil(len(all_fnames) / float(num_processes)))
begin_index = 0
indices = [(
                begin_index + i*num_each,
                min(begin_index + (i+1)*num_each,
                len(all_fnames))
           )
            for i in range(num_processes)]
print(indices)
inputs = [all_fnames[indices[i][0]:indices[i][1]] for i in range(num_processes)]
n_processed = sum(p.map(process_audios, inputs))
et = datetime.datetime.now()
print('Took: {} for {} files'.format(str(et-st), n_processed))

