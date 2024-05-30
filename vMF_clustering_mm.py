from composition.vMFMM import *
import models
import argparse
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pickle
import os
import logging

from models.crosscompcsd import CrossCSDFirst
from mmwhs_dataloader import MMWHS_single

def get_args():
	parser = argparse.ArgumentParser() 
	parser.add_argument('--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
	parser.add_argument('--name', type=str, default='test_cross_onecl_clip/pretrain/', help='The name of the checkpoints.')
	parser.add_argument('--data_dir',  type=str, default='../data/other/CT_withGT_proc/annotated/', help='Data direction target.')
	parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
	parser.add_argument('--norm', type=str, default="Batch")
	parser.add_argument('--k1', type=int,  default=40, help='When the learning rate starts decaying')
	parser.add_argument('--k2', type=int,  default=4, help='Check decay learning')
	parser.add_argument('--layer', type=int,  default=8, help='layer from which the deep features are obtained')
	parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
	parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')
	parser.add_argument('--cases_folds', type=int,  default=0, help='Cases folds')
	return parser.parse_args()


def main(args):
	######################################################################################
	###################################### load the extractor
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dir_instance = os.path.join(args.cp, args.name)

	model = CrossCSDFirst(args, device, 1)
	model_name = glob.glob(f"{dir_instance}/*.pth")[0]
	print(f'Model name: {model_name}')
	model.resume(model_name)
	model.to(device)
	model.eval()

	######################################################################################
	# Setup work
	###################################### change the directories
	dict_dir = os.path.join(dir_instance, 'kernels_true/')
	os.makedirs(dict_dir, exist_ok=True)

	vMF_kappa = 30 # kernel variance

	###################################### calculate the numbers
	# [y1, y2, y3, y4]
	Arf = 1 # ?????
	Apad = 0
	offset = 3
	######################################################################################

	total_images = 10000 # number of image for vMF kernels learning --> we only have 288
	samp_size_per_img = 15000 # number of positions in hxw --> they had 500 --> 5.000.000 in total, we have 4.320.000

	########################################### load the train images
	if args.cases_folds == 0:
		cases = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19]
	elif args.cases_folds == 1:
		cases = [0,1,2,4,6,7,9,10,12,13,14,15,16,17,18,19]
	elif args.cases_folds == 2:
		cases = [0,1,3,4,5,6,7,8,9,10,11,12,14,15,17,19]
	elif args.cases_folds == 3:
		cases = [0,1,2,3,5,6,7,8,10,11,13,14,15,16,17,18]
	elif args.cases_folds == 4:
		cases = [0,1,2,3,4,5,8,9,11,12,13,15,16,17,18,19]

	dataset_train = MMWHS_single(args, cases)
	train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
	######################################################################################


	loc_set = []
	feat_set = []
	imgs_list = []
	nfeats = 0
	for ii, data in enumerate(train_loader):
		input, _ = data

		if ii < total_images:
		# extract the features using the extractor
			with torch.no_grad():
				input = input.to(device)
				tmp = model.extract_feats_true(input).squeeze(0).detach().cpu().numpy() # 64, 256, 256

			# feature height and width
			height, width = tmp.shape[1:3]

			# trunk the features by cutting the outter 3 pixels
			tmp = tmp[:,offset:height - offset, offset:width - offset] # 64, 250, 250
			
			# dxhxw -> dxhw, d is the number of channels
			gtmp = tmp.reshape(tmp.shape[0], -1) # 64, 62500

			# randomly sample 500 feature vectors per sample
			if gtmp.shape[1] >= samp_size_per_img:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
			else:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]

			
			# transpose the feature vectors, now the dimension is nxd, n is 20
			tmp_feats = gtmp[:, rand_idx].T # 15000, 64

			cnt = 0
			for rr in rand_idx:
				# find the localization of the feature vector
				ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
				# original localization of feature vector -> ihi+offset
				# input.shape[2]/height -> downsampling scale
				# Apad -> number of padding pixels
				hi = (ihi+offset)*(input.shape[2]/height)-Apad
				wi = (iwi + offset)*(input.shape[3]/width)-Apad
				# save the localization of category of the image, index of the image, receptive field of the feature vector
				loc_set.append([ii, hi,wi,hi+Arf,wi+Arf]) # index of image, x,y and x+arf,y+arf
				# list of all feature vectors
				feat_set.append(tmp_feats[cnt,:])
				cnt+=1
			imgs_list.append(input.squeeze(0).squeeze(0).detach().cpu().numpy())

	feat_set = np.asarray(feat_set) # images*15000, 64
	loc_set = np.asarray(loc_set).T # 64, images*15000

	model = vMFMM(args.vc_num, 'k++')
	model.fit(feat_set, vMF_kappa, max_it=150)

	# save the initialized mu, nxk, dict_dir = kernels/init_unet/
	with open(dict_dir+'dictionary_{}.pickle'.format(args.vc_num), 'wb') as fh:
		pickle.dump(model.mu, fh)
	
	# model.mu == [12, 64]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)

