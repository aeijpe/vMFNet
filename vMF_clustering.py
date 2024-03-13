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

from models.unet_model import UNet
from mmwhs_dataloader import MMWHS_single

def get_args():
    usage_text = (
        "vMF clustering Pytorch Implementation"
        "Usage:  python pretrain.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-c', '--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
    parser.add_argument('-t', '--tv', type=str, default='A', help='The name of the checkpoints.')
    parser.add_argument('-mn', '--model_name', type=str, default='unet', help='Name of the model architecture to be used for training/testing.')

    parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')

    return parser.parse_args()


def main(args):

	######################################################################################
	###################################### load the extractor
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dir_checkpoint = os.path.join(args.cp, "encoder")



	extractor = UNet(n_classes=1)
	extractor.load_state_dict(torch.load(dir_checkpoint+'UNet.pth', map_location=device))
	extractor.to(device)
	extractor.eval()

	######################################################################################
	# Setup work
	###################################### change the directories
	kernels_save_dir = os.path.join(args.cp, 'kernels')

	vMF_kappa= 30 # kernel variance

	# initialization save directory
	# init_path = 'A_kernels/init_unet/'
	init_path = os.path.join(kernels_save_dir, 'init/')
	if not os.path.exists(init_path):
		os.makedirs(init_path)

	# dict_dir = 'A_kernels/init_unet/dictionary_unet/'
	dict_dir = os.path.join(init_path, 'dictionary/')
	if not os.path.exists(dict_dir):
		os.makedirs(dict_dir)

	###################################### calculate the numbers
	# [y1, y2, y3, y4]
	Arf_set = [8, 4, 2, 1]  # receptive field size
	Apad_set = [0, 0, 0, 0]  # padding size

	layer = 9 # [BS, 64, 256, 256]
	Arf = 1 # ?????
	Apad = 0
	offset = 3
	######################################################################################

	total_images = 10000 # number of image for vMF kernels learning
	samp_size_per_img = 500 # number of positions in hxw

	########################################### load the images, all images are for one label
	cases = range(0,18)
	dataset_train = MMWHS_single(args, cases)
	train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
	n_train = dataset_train.__len__()
	######################################################################################


	loc_set = []
	feat_set = []
	imgs_list = []
	nfeats = 0
	for ii,data in enumerate(train_loader):
		input, _ = data
		if np.mod(ii,500)==0:
			print('{} / {}'.format(ii,n_train))

		if ii < total_images:
		# extract the features using the extractor
			with torch.no_grad():
				tmp = extractor(input.to(device))[layer].squeeze(0).detach().cpu().numpy() 

			print("tmp shape: ", tmp.shape)

			# feature height and width
			height, width = tmp.shape[1:3]
			print("height: ", height)
			print("width: ", width)

			# trunk the features by cutting the outter 3 pixels
			tmp = tmp[:,offset:height - offset, offset:width - offset]
			# dxhxw -> dxhw, d is the number of channels
			gtmp = tmp.reshape(tmp.shape[0], -1) # 128, 72x72
			# randomly sample 20 feature vector per sample
			if gtmp.shape[1] >= samp_size_per_img:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
			else:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
					#rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
			# transpose the feature vectors, now the dimension is nxd, n is 20
			tmp_feats = gtmp[:, rand_idx].T # 1000, 128

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

	feat_set = np.asarray(feat_set)
	loc_set = np.asarray(loc_set).T

	print(feat_set.shape) # 2562000, 128
	print(loc_set.shape) # 5, 2562000
	model = vMFMM(args.vc_num, 'k++')
	model.fit(feat_set, vMF_kappa, max_it=150)

	# save the initialized mu, nxk, dict_dir = kernels/init_unet/
	with open(dict_dir+'dictionary_{}.pickle'.format(args.vc_num), 'wb') as fh:
		pickle.dump(model.mu, fh)

	##################################### work on the following

	num = 50
	SORTED_IDX = []
	SORTED_LOC = []
	# vc_i -> 0 - 511
	for vc_i in range(args.vc_num):
	# p: nxk
	# for every kernel, get the 50 feature vectors with minimal distances
	sort_idx = np.argsort(-model.p[:, vc_i])[0:num]
	SORTED_IDX.append(sort_idx)
	tmp=[]
	for idx in range(num):
		###################################### change here to extract the localizations
		iloc = loc_set[:, sort_idx[idx]]
		tmp.append(iloc)
	# get the localization of receptive field for the 50 feature vectors for each kernel
	SORTED_LOC.append(tmp)

	# save the distances, too large, more than 4Gb
	# with open(dict_dir + 'dictionary_{}_p.pickle'.format(vc_num), 'wb') as fh:
	# 	pickle.dump(model.p, fh)
	# p = model.p

	print('save top {0} images for each cluster'.format(num))
	example = [None for vc_i in range(vc_num)] # 512 None
	out_dir = dict_dir + '/cluster_images_{}/'.format(vc_num) # 512 forlders
	if not os.path.exists(out_dir):
	os.makedirs(out_dir)

	print('')

	# save the 50 images for each kernel
	for vc_i in range(vc_num):
	# receptive field**2 and 1 channels, 50 images
	patch_set = np.zeros(((Arf**2)*1, num)).astype('uint8')
	# index for each kernel
	sort_idx = SORTED_IDX[vc_i]#np.argsort(-p[:,vc_i])[0:num]
	opath = out_dir + str(vc_i) + '/'
	if not os.path.exists(opath):
		os.makedirs(opath)
	locs=[]
	for idx in range(num):
		iloc = loc_set[:,sort_idx[idx]]
		loc = iloc[0:5].astype(int)
		if not loc[0] in locs:
			locs.append(loc[0])
			# img = cv2.imread(imgs[int(loc[0])])
			img = imgs_list[int(loc[0])]
			img *= 255
			patch = img[loc[1]:loc[3], loc[2]:loc[4]]
			#patch_set[:,idx] = patch.flatten()
			if patch.size:
				cv2.imwrite(opath+str(idx)+'.JPEG',patch)
	#example[vc_i] = np.copy(patch_set)
	if vc_i%10 == 0:
		print(vc_i)

	# print summary for each vc
	#if layer=='pool4' or layer =='last': # somehow the patches seem too big for p5
	for c in range(vc_num):
	iidir = out_dir + str(c) +'/'
	files = glob.glob(iidir+'*.JPEG')
	width = 100
	height = 100
	canvas = np.zeros((0,4*width,3))
	cnt = 0
	for jj in range(4):
		row = np.zeros((height,0,3))
		ii=0
		tries=0
		next=False
		for ii in range(4):
			if (jj*4+ii)< len(files):
				img_file = files[jj*4+ii]
				if os.path.exists(img_file):
					img = cv2.imread(img_file)
				img = cv2.resize(img, (width,height))
			else:
				img = np.zeros((height, width, 3))
			row = np.concatenate((row, img), axis=1)
		canvas = np.concatenate((canvas,row),axis=0)

	cv2.imwrite(out_dir+str(c)+'.JPEG',canvas)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)

