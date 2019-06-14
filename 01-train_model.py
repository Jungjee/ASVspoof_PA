from comet_ml import Experiment
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
#from torchsummary import summary

from spec_CNN import spec_CNN

def summary(model, input_size, batch_size=-1, device="cuda", print_fn = None):
	if print_fn == None: printfn = print

	def register_hook(module):

		def hook(module, input, output):
			class_name = str(module.__class__).split(".")[-1].split("'")[0]
			module_idx = len(summary)

			m_key = "%s-%i" % (class_name, module_idx + 1)
			summary[m_key] = OrderedDict()
			summary[m_key]["input_shape"] = list(input[0].size())
			summary[m_key]["input_shape"][0] = batch_size
			if isinstance(output, (list, tuple)):
				summary[m_key]["output_shape"] = [
					[-1] + list(o.size())[1:] for o in output
				]
			else:
				summary[m_key]["output_shape"] = list(output.size())
				summary[m_key]["output_shape"][0] = batch_size

			params = 0
			if hasattr(module, "weight") and hasattr(module.weight, "size"):
				params += torch.prod(torch.LongTensor(list(module.weight.size())))
				summary[m_key]["trainable"] = module.weight.requires_grad
			if hasattr(module, "bias") and hasattr(module.bias, "size"):
				params += torch.prod(torch.LongTensor(list(module.bias.size())))
			summary[m_key]["nb_params"] = params

		if (
			not isinstance(module, nn.Sequential)
			and not isinstance(module, nn.ModuleList)
			and not (module == model)
		):
			hooks.append(module.register_forward_hook(hook))

	device = device.lower()
	assert device in [
		"cuda",
		"cpu",
	], "Input device is not valid, please specify 'cuda' or 'cpu'"

	if device == "cuda" and torch.cuda.is_available():
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	# multiple inputs to the network
	if isinstance(input_size, tuple):
		input_size = [input_size]

	# batch_size of 2 for batchnorm
	x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
	# print(type(x[0]))

	# create properties
	summary = OrderedDict()
	hooks = []

	# register hook
	model.apply(register_hook)

	# make a forward pass
	# print(x.shape)
	model(*x)

	# remove these hooks
	for h in hooks:
		h.remove()

	print_fn("----------------------------------------------------------------")
	line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
	print_fn(line_new)
	print_fn("================================================================")
	total_params = 0
	total_output = 0
	trainable_params = 0
	for layer in summary:
		# input_shape, output_shape, trainable, nb_params
		line_new = "{:>20}  {:>25} {:>15}".format(
			layer,
			str(summary[layer]["output_shape"]),
			"{0:,}".format(summary[layer]["nb_params"]),
		)
		total_params += summary[layer]["nb_params"]
		total_output += np.prod(summary[layer]["output_shape"])
		if "trainable" in summary[layer]:
			if summary[layer]["trainable"] == True:
				trainable_params += summary[layer]["nb_params"]
		print_fn(line_new)

	# assume 4 bytes/number (float on cuda).
	total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
	total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
	total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
	total_size = total_params_size + total_output_size + total_input_size

	print_fn("================================================================")
	print_fn("Total params: {0:,}".format(total_params))
	print_fn("Trainable params: {0:,}".format(trainable_params))
	print_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
	print_fn("----------------------------------------------------------------")
	print_fn("Input size (MB): %0.2f" % total_input_size)
	print_fn("Forward/backward pass size (MB): %0.2f" % total_output_size)
	print_fn("Params size (MB): %0.2f" % total_params_size)
	print_fn("Estimated Total Size (MB): %0.2f" % total_size)
	print_fn("----------------------------------------------------------------")


def balance_classes(lines_small, lines_big, np_seed):
	'''
	Balance number of sample per class.
	Designed for two-class classification.
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
	l_gen = []
	l_spo = []
	for r, ds, fs in os.walk(src_dir):
		for f in fs:
			if f[-2:] != 'pt':
				continue
			k = f.split('.')[0]
			if k[-1] == '1':
				l_gen.append(k)
			else:
				l_spo.append(k)

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

		X = torch.load(self.base_dir + ID + '.pt')

		#print(X, type(X))
		#print(X.size())
		nb_time = X.shape[0]
		if nb_time > self.nb_time:
			start_idx = np.random.randint(low = 0,
				high = nb_time - self.nb_time)
			#X = X[start_idx:start_idx+self.nb_time, :, :].permute(2, 0, 1)
			X = X[start_idx:start_idx+self.nb_time, :, :].transpose(2, 0, 1)
		elif nb_time < self.nb_time:
			nb_dup = int(self.nb_time / nb_time) + 1
			#X = X.repeat(nb_dup, 1, 1)[start_idx:start_idx+self.nb_time, :, :].permute(2, 0, 1)
			X = np.tile(X, (nb_dup, 1, 1))[:self.nb_time, :, :].transpose(2, 0, 1)
		else:
			X = X.transpose(2, 0, 1)
			
		y = self.labels[ID]

		return X, y
	

if __name__ == '__main__':
	#load yaml file & set comet_ml config
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)
	experiment = Experiment(api_key="9CueLwB3ujfFlhdD9Z2VpKKaq",
		project_name="dcase2019", workspace="jungjee",
		disabled = bool(parser['comet_disable']))
	experiment.set_name(parser['name'])
	
	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

	#get 4 utt_lists
	l_gen_trn, l_spo_trn= get_utt_list(parser['DB'] + 'spectrogram_trn/')
	l_gen_dev, l_spo_dev= get_utt_list(parser['DB'] + 'spectrogram_dev/')
	
	#define labels
	d_label_trn = {}
	d_label_dev = {}
	for key in l_gen_trn:
		d_label_trn[key] = 1
	for key in l_spo_trn:
		d_label_trn[key] = 0
	for key in l_gen_dev:
		d_label_dev[key] = 1
	for key in l_spo_dev:
		d_label_dev[key] = 0

	#get balanced validation utterance list.
	l_dev_utt = balance_classes(l_gen_dev, l_spo_dev, 0)

	#define dataset generators
	devset = Dataset_ASVspoof2019_PA(list_IDs = l_dev_utt,
		labels = d_label_dev,
		nb_time = parser['nb_time'],
		base_dir = parser['DB'] + '/spectrogram_dev/')
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = False,
		num_workers = parser['nb_proc_db'])

	#set save directory
	save_dir = parser['save_dir'] + parser['name'] + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir  + 'results_cnn/'):
		os.makedirs(save_dir + 'results_cnn/')
	if not os.path.exists(save_dir  + 'cnn_models/'):
		os.makedirs(save_dir + 'cnn_models/')
	
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
	model = spec_CNN(parser['model']).to(device)

	#'''
	#log model summary to file
	with open(save_dir + 'summary_cnn.txt', 'w+') as f_summary:
		#summ = summary(model, input_size = (parser['model']['in_channels'], parser['nb_time'], parser['feat_dim'])) # configure input_size as (channels, H, W)
		summary(model,
			input_size = (parser['model']['in_channels'], parser['nb_time'], parser['feat_dim']),
			print_fn=lambda x: f_summary.write(x + '\n')) # configure input_size as (channels, H, W)

	if len(parser['gpu_idx']) > 1:
		model = nn.DataParallel(model, device_ids = parser['gpu_idx'])
	#'''

	#set ojbective funtion
	criterion = nn.CrossEntropyLoss()

	#set optimizer
	if parser['optimizer'].lower() == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(),
			lr = parser['lr'],
			momentum = parser['opt_mom'],
			weight_decay = parser['wd'],
			nesterov = bool(parser['nesterov']))

	elif parser['optimizer'].lower() == 'adam':
		optimizer = torch.optim.Adam(model.parameters(),
			lr = parser['lr'],
			weight_decay = parser['wd'],
			amsgrad = bool(parser['amsgrad']))

	##########################################
	#train/val################################
	##########################################
	best_cnn_eer = 99.
	f_eer = open(save_dir + 'eers_cnn.txt', 'a', buffering = 1)
	for epoch in tqdm(range(parser['epoch'])):
		f_eer.write('%d '%epoch)

		#make classwise-balanced utt list for this epoch.
		#def balance_classes(lines_small, lines_big, np_seed):
		trn_list_cur = balance_classes(l_gen_trn, l_spo_trn, int(epoch))

		#define dataset generators
		trnset = Dataset_ASVspoof2019_PA(list_IDs = trn_list_cur,
			labels = d_label_trn,
			nb_time = parser['nb_time'],
			base_dir = parser['DB'] + '/spectrogram_trn/')
		trnset_gen = data.DataLoader(trnset,
			batch_size = parser['batch_size'],
			shuffle = True,
			num_workers = parser['nb_proc_db'])

		#train phase
		with tqdm(total = len(trnset_gen)) as pbar:
			for m_batch, m_label in trnset_gen:
				#continue #temporary
				m_batch, m_label = m_batch.to(device), m_label.to(device)
	
				output = model(m_batch)
				optimizer.zero_grad()
				loss = criterion(output, m_label)
				#print(loss)
				loss.backward()
				optimizer.step()
				pbar.set_description('epoch: %d loss: %.3f'%(epoch, loss))
				pbar.update(1)
		experiment.log_metric('loss', loss)

		#validate only odd epochs for speed-up.
		if epoch % 2 == 1: continue

		#validation phase
		with torch.set_grad_enabled(False):
			with tqdm(total = len(devset_gen)) as pbar:
				y_score = [] # score for each sample
				y = [] # label for each sample 
				for m_batch, m_label in devset_gen:
					m_batch, m_labels = m_batch.to(device), m_label.to(device)
					y.extend(list(m_label))
					#y_score = model(m_batch).cpu().numpy() #>>> (16, 64?)
					y_score.extend(model(m_batch).cpu().numpy()[:,0]) #>>> (16, 64?)
					#print(y_score)
					pbar.update(1)
			#print(np.array(y_score).shape)
			
			#calculate EER
			#y, y_score = eval_softmax(val_lines, model_cnn_pred, base_dir = parser['base_dir'])
			f_res = open(save_dir + 'results_cnn/epoch%s.txt'%(epoch), 'w')
			for _s, _t in zip(y, y_score):
				f_res.write('{score} {target}\n'.format(score=_s,target=_t))
			f_res.close()
			fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=0)
			eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
			print(eer)
			experiment.log_metric('val_eer', eer)
			f_eer.write('%f \n'%eer)
	
			#record best validation model
			if float(eer) < best_cnn_eer:
				print('New best EER: %f'%float(eer))
				best_cnn_eer = float(eer)
				dir_best_cnn_model_weights = save_dir + 'cnn_models/%d-%.6f.h5'%(epoch, eer)
				experiment.log_metric('best_val_eer', eer)
				
				#save best model
				if len(parser['gpu_idx']) > 1: # multi GPUs
					torch.save(model.module.state_dict(), save_dir +  'cnn_models/best.pt')
				else: #single GPU
					torch.save(model.state_dict(), save_dir +  'cnn_models/best.pt')
				
			if not bool(parser['save_best_only']):
				#save model
				if len(parser['gpu_idx']) > 1: # multi GPUs
					torch.save(model.module.state_dict(), save_dir +  'cnn_models/%d-%.6f.pt'%(epoch, eer))
				else: #single GPU
					torch.save(model.state_dict(), save_dir +  'cnn_models/%d-%.6f.pt'%(epoch, eer))
				
	f_eer.close()

















