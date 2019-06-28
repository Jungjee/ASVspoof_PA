import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch.utils import data
from torch.nn.parameter import Parameter
from collections import OrderedDict

class Residual_block(nn.Module):
	def __init__(self, nb_filts, kernels, strides, first = False, downsample = False):
		super(Residual_block, self).__init__()
		self.first = first
		self.downsample = downsample

		if not self.first:
			self.bn1 = nn.BatchNorm2d(num_features = nb_filts[0])

		#self.lrelu = nn.LeakyReLU(negative_slope = 0.3)
		self.lrelu = nn.LeakyReLU()
		self.conv1 = nn.Conv2d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = kernels,
			padding = (1, 3),
			stride = strides)
		self.bn2 = nn.BatchNorm2d(num_features = nb_filts[1])
		self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = (1, 3),
			kernel_size = kernels,
			stride = 1)

		if downsample:
			self.conv_downsample = nn.Conv2d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = (1, 3),
				kernel_size = kernels,
				stride = strides)
			#self.bn_downsample = nn.BatchNorm2d(num_features = nb_filts[2])


	def forward(self, x):
		identity = x

		if not self.first:
			out = self.bn1(x)
			out = self.lrelu(out)
		else:
			out = x

		out = self.conv1(out)
		out = self.bn2(out)
		out = self.lrelu(out)
		out = self.conv2(out)

		if self.downsample:
			identity = self.conv_downsample(identity)
			#identity = self.bn_downsample(identity)
		
		out += identity
		#print(identity.size())
		return out


class spec_CNN_GRU(nn.Module):
	#def __init__(self, nb_filts, kernels, strides, name, first = False, downsample = False):
	def __init__(self, d_args, device, do_pretrn = True):
		super(spec_CNN_GRU, self).__init__()
		self.device = device		#for center loss
		self.do_pretrn = do_pretrn
		self.forward_mode = 'cnn' if self.do_pretrn else 'gru'

		self.first_conv = nn.Conv2d(in_channels = d_args['in_channels'],
			out_channels = d_args['filts'][0],
			kernel_size = d_args['kernels'][0],
			padding = (1, 3),
			stride = d_args['strides'][0])
		self.first_bn = nn.BatchNorm2d(num_features = d_args['filts'][0])
		#self.first_lrelu = nn.LeakyReLU(negative_slope = 0.3)
		self.first_lrelu = nn.LeakyReLU()

		self.block0 = Residual_block(nb_filts = d_args['filts'][1],
			kernels = d_args['kernels'][1],
			strides = d_args['strides'][1],
			first = True,
			downsample = True)

		self.block1 = self._make_layer(nb_blocks = d_args['blocks'][0],
			nb_filts = d_args['filts'][2],
			kernels = d_args['kernels'][2],
			strides = d_args['strides'][2])

		self.block2 = self._make_layer(nb_blocks = d_args['blocks'][1],
			nb_filts = d_args['filts'][3],
			kernels = d_args['kernels'][3],
			strides = d_args['strides'][3])

		self.block3 = self._make_layer(nb_blocks = d_args['blocks'][2],
			nb_filts = d_args['filts'][4],
			kernels = d_args['kernels'][4],
			strides = d_args['strides'][4])

		self.block4 = self._make_layer(nb_blocks = d_args['blocks'][3],
			nb_filts = d_args['filts'][5],
			kernels = d_args['kernels'][5],
			strides = d_args['strides'][5],
			downsample = False)

		self.last_bn = nn.BatchNorm2d(num_features = d_args['filts'][-1][-1])
		#self.last_lrelu = nn.LeakyReLU(negative_slope = 0.3)
		self.last_lrelu = nn.LeakyReLU()

		#self.lrelu = nn.LeakyReLU(negative_slope = 0.3)
		self.lrelu = nn.LeakyReLU()
		if do_pretrn:
			self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
			self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
			self.fc1_cnn = nn.Linear(in_features = d_args['filts'][-1][-1] * 2,
				out_features = d_args['nb_fc_node'])
			self.fc2_cnn = nn.Linear(in_features = d_args['nb_fc_node'],
				out_features = d_args['nb_classes'],
				bias = False)

		#shape: (batch, channel, time, freq)
		self.maxpool_gru = nn.MaxPool2d((1, 17))	#need to be done adaptive to input size
		self.avgpool_gru = nn.AvgPool2d((1, 17))	#need to be done adaptive to input size
		self.gru = nn.GRU(input_size = d_args['filts'][-1][-1] * 2,
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)
		self.fc1_gru = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_fc_node'])
		self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],
			bias = False)

	def _make_layer(self, nb_blocks, nb_filts, kernels, strides, downsample = True):
		layers = []
		for _ in range(nb_blocks - 1):
			layers.append(Residual_block(nb_filts = [nb_filts[0], nb_filts[0]],
				kernels = kernels,
				strides = 1))
		if downsample:
			layers.append(Residual_block(nb_filts = nb_filts,
				kernels = kernels,
				strides = strides,
				downsample = downsample))

		return nn.Sequential(*layers)

		
	def forward(self, x):
		x = self.first_conv(x)
		x = self.first_bn(x)
		x = self.first_lrelu(x)

		x = self.block0(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		x = self.last_bn(x)
		x = self.last_lrelu(x)

		if self.forward_mode == 'cnn':
			x_avg = self.global_avgpool(x)
			channel_dim = x_avg.size(1)
			x_avg = x_avg.view(-1, channel_dim)
			x_max = self.global_maxpool(x)
			x_max = x_max.view(-1, channel_dim)
			x = torch.cat((x_avg, x_max), dim = 1)
			#x = self.fc1_cnn(x)
			#code = self.lrelu(x)
			#out = self.fc2_cnn(code)
			code = self.fc1_cnn(x)
			out = self.fc2_cnn(code)

		elif self.forward_mode == 'gru':	
			x_avg = self.avgpool_gru(x)# should be (batch, channel, time, 1)
			_, channel_dim, time_dim, _ = x_avg.size()
			x_avg = x_avg.view(-1, channel_dim, time_dim).permute(0,2,1)# should be (batch, time, channel)
			x_max = self.maxpool_gru(x)
			x_max = x_max.view(-1, channel_dim, time_dim).permute(0,2,1)# should be (batch, time, channel)
			x = torch.cat((x_avg, x_max), dim = 2)
			x, _ = self.gru(x)
			x = x[:,-1,:]
			#x = self.fc1_gru(x)
			#code = self.lrelu(x)
			#out = self.fc2_gru(code)
			code = self.fc1_gru(x)
			out = self.fc2_gru(code)
		else:
			raise NotImplementedError('"forward mode" should be ["cnn", "gru"]')

		return code, out



	def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
		if print_fn == None: printfn = print
		model = self
	
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
					if len(summary[m_key]["output_shape"]) != 0:
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
		if isinstance(input_size, tuple):
			input_size = [input_size]
		x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
		summary = OrderedDict()
		hooks = []
		model.apply(register_hook)
		model(*x)
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
		return


