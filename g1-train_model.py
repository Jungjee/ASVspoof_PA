from comet_ml import Experiment
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

from spec_CNN_c_bc import spec_CNN


def balance_classes(lines_small, lines_big, np_seed):
	'''
	Balance number of sample per class.
	Designed for Binary(two-class) classification.
	'''

	len_small_lines = len(lines_small)
	len_big_lines = len(lines_big)
	idx_big = list(range(len_big_lines))

	np.random.seed(np_seed)
	np.random.shuffle(lines_big)
	new_lines = lines_small + lines_big[:len_small_lines]
	np.random.shuffle(new_lines)
	#print(new_lines[:5])

	return new_lines


def get_utt_list(src_dir):
	'''
	Designed for ASVspoof2019 PA
	'''
	l_utt = []
	for r, ds, fs in os.walk(src_dir):
		for f in fs:
			if f[-3:] != 'npy': continue
			l_utt.append(f.split('.')[0])

	return l_utt

def split_genSpoof(l_in, dir_meta, return_dic_meta = False):
	l_gen, l_spo = [], []
	d_meta = {}

	with open(dir_meta, 'r') as f:
		l_meta = f.readlines()
	for line in l_meta:
		_, key, _, _, label = line.strip().split(' ')
		d_meta[key] = 1 if label == 'bonafide' else 0

	for k in d_meta.keys():
		if d_meta[k] == 1:
			l_gen.append(k)
		else:
			l_spo.append(k)
		
	if return_dic_meta:
		return l_gen, l_spo, d_meta
	else:
		return l_gen, l_spo
			

class Dataset_ASVspoof2019_PA(data.Dataset):
	def __init__(self, list_IDs, labels, nb_time, base_dir):
		'''
		self.list_IDs	: list of strings (each string: utt key)
		self.labels		: dictionary (key: utt key, value: label integer)
		self.nb_time	: integer, the number of timesteps for each mini-batch
		'''
		self.list_IDs = list_IDs
		self.labels = labels
		self.nb_time = nb_time
		self.base_dir = base_dir


	def __len__(self):
		return len(self.list_IDs)


	def __getitem__(self, index):
		ID = self.list_IDs[index]
		X = np.load(self.base_dir + ID + '.npy')

		#print(X.shape)>>> (1, time, freq)
		#exit()
		nb_time = X.shape[1]
		if nb_time > self.nb_time:
			start_idx = np.random.randint(low = 0,
				high = nb_time - self.nb_time)
			X = X[:, start_idx:start_idx+self.nb_time, :]
		elif nb_time < self.nb_time:
			nb_dup = int(self.nb_time / nb_time) + 1
			X = np.tile(X, (1, nb_dup, 1))[:, :self.nb_time, :]
		#print(X.shape)
			
		y = self.labels[ID]

		return X, y
	

if __name__ == '__main__':
	#load yaml file & set comet_ml config
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)
	experiment = Experiment(api_key="9CueLwB3ujfFlhdD9Z2VpKKaq",
		project_name="torch_spoof19", workspace="jungjee",
		disabled = bool(parser['comet_disable']))
	experiment.set_name(parser['name'])
	
	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

	#get 4 utt_lists
	l_trn = get_utt_list(parser['DB'] + parser['DB_trn'])
	l_dev = get_utt_list(parser['DB'] + parser['DB_dev'])
	l_eval = get_utt_list(parser['DB'] + parser['DB_eval'])
	l_gen_trn, l_spo_trn, d_label_trn = split_genSpoof(l_in = l_trn, dir_meta = parser['DB']+parser['dir_meta_trn'], return_dic_meta = True)
	l_gen_dev, l_spo_dev, d_label_dev = split_genSpoof(l_in = l_dev, dir_meta = parser['DB']+parser['dir_meta_dev'], return_dic_meta = True)
	l_gen_eval, l_spo_eval, d_label_eval = split_genSpoof(l_in = l_eval, dir_meta = parser['DB']+parser['dir_meta_eval'], return_dic_meta = True)
	del l_trn, l_dev, l_eval
	
	#get balanced validation utterance list.
	l_dev_utt = balance_classes(l_gen_dev, l_spo_dev, 0)	#for speed-up only
	l_eval_utt = balance_classes(l_gen_eval, l_spo_eval, 0)	#for speed-up only
	del l_gen_dev, l_spo_dev, l_gen_eval, l_spo_eval

	#define dataset generators
	devset = Dataset_ASVspoof2019_PA(list_IDs = l_dev_utt,
		labels = d_label_dev,
		nb_time = parser['nb_time'],
		base_dir = parser['DB'] + parser['DB_dev'])
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])
	evalset = Dataset_ASVspoof2019_PA(list_IDs = l_eval_utt,
		labels = d_label_eval,
		nb_time = parser['nb_time'],
		base_dir = parser['DB'] + parser['DB_eval'])
	evalset_gen = data.DataLoader(evalset,
		batch_size = parser['batch_size'],
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])

	#set save directory
	save_dir = parser['save_dir'] + parser['name'] + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir  + 'results/'):
		os.makedirs(save_dir + 'results/')
	if not os.path.exists(save_dir  + 'models/'):
		os.makedirs(save_dir + 'models/')
	
	#log experiment parameters to local and comet_ml server
	#to local
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in parser.items():
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.close()

	#to comet server
	experiment.log_parameters(parser)
	experiment.log_parameters(parser['model'])

	#define model
	model = spec_CNN(parser['model'], device).to(device)

	#log model summary to file
	with open(save_dir + 'summary.txt', 'w+') as f_summary:
		model.summary(
			input_size = (parser['model']['in_channels'], parser['nb_time'], parser['feat_dim']),
			print_fn = lambda x: f_summary.write(x + '\n')) # configure input_size as (channels, H, W)

	if len(parser['gpu_idx']) > 1:
		model = nn.DataParallel(model, device_ids = parser['gpu_idx'])

	#set ojbective funtions
	criterion = nn.CrossEntropyLoss()

	#set optimizer
	params = list(model.parameters())
	if parser['optimizer'].lower() == 'sgd':
		optimizer = torch.optim.SGD(params,
			lr = parser['lr'],
			momentum = parser['opt_mom'],
			weight_decay = parser['wd'],
			nesterov = bool(parser['nesterov']))

	elif parser['optimizer'].lower() == 'adam':
		optimizer = torch.optim.Adam(params,
			lr = parser['lr'],
			weight_decay = parser['wd'],
			amsgrad = bool(parser['amsgrad']))

	##########################################
	#train/val################################
	##########################################
	best_eer = 99.
	f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)
	for epoch in tqdm(range(parser['epoch'])):
		f_eer.write('%d '%epoch)

		#make classwise-balanced utt list for this epoch.
		#def balance_classes(lines_small, lines_big, np_seed):
		trn_list_cur = balance_classes(l_gen_trn, l_spo_trn, int(epoch))

		#define dataset generators
		trnset = Dataset_ASVspoof2019_PA(list_IDs = trn_list_cur,
			labels = d_label_trn,
			nb_time = parser['nb_time'],
		base_dir = parser['DB'] + parser['DB_trn'])
		trnset_gen = data.DataLoader(trnset,
			batch_size = parser['batch_size'],
			shuffle = True,
			drop_last = True,
			num_workers = parser['nb_proc_db'])

		#train phase
		model.train()
		with tqdm(total = len(trnset_gen), ncols = 70) as pbar:
			for m_batch, m_label in trnset_gen:
				m_batch, m_label = m_batch.to(device), m_label.to(device)
	
				code, output = model(m_batch)
				cce_loss = criterion(output, m_label)
				c_loss = model.compute_center_loss(code, model.centers, m_label)
				loss = cce_loss + (parser['c_loss_weight'] * c_loss)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				c_deltas = model.get_center_delta(code.data, model.centers, m_label, parser['c_loss_lr'])
				model.centers = model.centers - c_deltas
				pbar.set_description('epoch%d:\tloss_cce:\t%.3f\tloss_c:%.3f'%(epoch, cce_loss, c_loss))
				pbar.update(1)
		experiment.log_metric('loss', loss)

		#validate only odd epochs for speed-up.
		#if epoch % 2 == 1: continue

		#validation phase
		model.eval()
		with torch.set_grad_enabled(False):
			with tqdm(total = len(devset_gen), ncols = 70) as pbar:
				y_score = [] # score for each sample
				y = [] # label for each sample 
				for m_batch, m_label in devset_gen:
					m_batch, m_labels = m_batch.to(device), m_label.to(device)
					y.extend(list(m_label))
					_, out = model(m_batch)
					y_score.extend(out.cpu().numpy()[:,0]) #>>> (16, 64?)
					pbar.update(1)
			
			#calculate EER
			f_res = open(save_dir + 'results/epoch%s.txt'%(epoch), 'w')
			for _s, _t in zip(y, y_score):
				f_res.write('{score} {target}\n'.format(score=_s,target=_t))
			f_res.close()
			fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=0)
			eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
			print(eer)
			experiment.log_metric('val_eer', eer)
			f_eer.write('%f \n'%eer)
	
			#record best validation model
			if float(eer) < best_eer:
				print('New best EER: %f'%float(eer))
				best_eer = float(eer)
				dir_best_model_weights = save_dir + 'models/%d-%.6f.h5'%(epoch, eer)
				experiment.log_metric('best_val_eer', eer)
				
				#save best model
				if len(parser['gpu_idx']) > 1: # multi GPUs
					torch.save(model.module.state_dict(), save_dir +  'models/best.pt')
				else: #single GPU
					torch.save(model.state_dict(), save_dir +  'models/best.pt')
				
			if not bool(parser['save_best_only']):
				#save model
				if len(parser['gpu_idx']) > 1: # multi GPUs
					torch.save(model.module.state_dict(), save_dir +  'models/%d-%.6f.pt'%(epoch, eer))
				else: #single GPU
					torch.save(model.state_dict(), save_dir +  'models/%d-%.6f.pt'%(epoch, eer))
				
	f_eer.close()

















