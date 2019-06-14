# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import os

from scipy.signal import spectrogram
from multiprocessing import Process

def pre_emp(x):
	'''
	Apply pre-emphasis to given utterance.
	x	: list or 1 dimensional numpy.ndarray
	'''
	return np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32)

def extract_spectrograms(l_utt):
	'''	
	Extracts spectrograms
	'''
	for line in l_utt:
		utt, _ = sf.read(line, dtype = 'int16')
		utt = pre_emp(utt)

		_, _, spec = spectrogram(x = utt,
			fs = _fs,
			window = _window,
			nperseg = _nperseg,
			noverlap = _noverlap,
			nfft = _nfft,
			mode = _mode)
		spec = np.expand_dims(spec.T, axis = 0).astype(np.float32)		# add 0 dim for torch

		dir_base, fn = os.path.split(line)
		dir_base, _ = os.path.split(dir_base)
		fn, _ = os.path.splitext(fn) 
		if not os.path.exists(dir_base + _dir_name):
			os.makedirs(dir_base + _dir_name)
		np.save(dir_base+_dir_name+fn, spec)
	return 
#======================================================================#
_nb_proc = 24									# numer of sub-processes (set 1 for single process)
_fs = 16000										# sampling rate
_window = 'hamming'								# window type
_mode = 'magnitude'								# [psd, complex, magnitude, angle, phase]
_nfft = 2048									# number of fft bins
_nperseg = int(50 * _fs * 0.001)				# window length (in ms)
_noverlap = int(30 * _fs * 0.001)				# window shift size (in ms)
_dir_dataset = '/home/leo/DB/ASVspoof2019/'		# directory of Dataset
_dir_name = '/spec_{}_{}_{}_{}/'.format(_mode, _nfft, _nperseg, _noverlap)

if __name__ == '__main__':
	l_utt = []
	for r, ds, fs in os.walk(_dir_dataset):
		for f in fs:
			if os.path.splitext(f)[1] != '.flac': continue
			l_utt.append('/'.join([r, f.replace('\\', '/')]))
	
	nb_utt_per_proc = int(len(l_utt) / _nb_proc)
	l_proc = []
	for i in range(_nb_proc):
		if i == _nb_proc - 1:
			l_utt_cur = l_utt[i * nb_utt_per_proc :]
		else:
			l_utt_cur = l_utt[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]
		l_proc.append(Process(target = extract_spectrograms, args = (l_utt_cur,)))
		print('%d'%i)

	for i in range(_nb_proc):
		l_proc[i].start()
		print('start %d'%i)
	for i in range(_nb_proc):
		l_proc[i].join()









