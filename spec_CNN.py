import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

class Residual_block(nn.Module):
	def __init__(self, nb_filts, kernels, strides, first = False, downsample = False):
		super(Residual_block, self).__init__()
		self.first = first
		self.downsample = downsample

		if not self.first:
			self.bn1 = nn.BatchNorm2d(num_features = nb_filts[0])

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


class CenterLoss(nn.Module):
	"""Center loss.
	
	Reference:
	Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
	
	Args:
		num_classes (int): number of classes.
		feat_dim (int): feature dimension.
	"""
	def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu

		if self.use_gpu:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long()
		if self.use_gpu: classes = classes.cuda()
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = distmat * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

		return loss

		

class spec_CNN(nn.Module):
	#def __init__(self, nb_filts, kernels, strides, name, first = False, downsample = False):
	def __init__(self, d_args):
		super(spec_CNN, self).__init__()
		self.first_conv = nn.Conv2d(in_channels = d_args['in_channels'],
			out_channels = d_args['filts'][0],
			kernel_size = d_args['kernels'][0],
			padding = (1, 3),
			stride = d_args['strides'][0])
		self.first_bn = nn.BatchNorm2d(num_features = d_args['filts'][0])
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
		self.last_lrelu = nn.LeakyReLU()
		self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
		self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.fc1 = nn.Linear(in_features = d_args['filts'][-1][-1] * 2,
			out_features = d_args['nb_fc_node'])
		self.fc2 = nn.Linear(in_features = d_args['nb_fc_node'],
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
		
		x_avg = self.global_avgpool(x)
		channel_dim = x_avg.size(1)
		x_avg = x_avg.view(-1, channel_dim)
		x_max = self.global_maxpool(x)
		x_max = x_max.view(-1, channel_dim)
		x = torch.cat((x_avg, x_max), dim = 1)

		x = self.fc1(x)
		x = self.fc2(x)


		return x


















