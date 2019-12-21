# coding: utf-8
import torch
import torch.nn as nn
from models.resnet_coral import *

import cv2  # read image
import numpy as np

def load_image(img_path):
	img = cv2.imread(img_path)
	if img is None:
		return None
	resized = cv2.resize(img, (112, 112))
	ccropped = resized[...,::-1] # BGR to RGB
	flipped = cv2.flip(ccropped, 1)
	ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
	ccropped = np.reshape(ccropped, [1, 3, 112, 112])
	ccropped = np.array(ccropped, dtype = np.float32)
	ccropped = (ccropped - 127.5) / 128.0
	ccropped = torch.from_numpy(ccropped)
	return ccropped

if __name__ == '__main__':
	DEVICE = torch.device('cuda:5')
	model = resnet18(num_classes=100, grayscale=False)
	model.to(DEVICE)
	model.load_state_dict(torch.load('checkpoints/coral_age_epoch_33.pth'))
	model.eval()

	image = load_image('54532-0.jpg')
	if image is None:
		print('input image is none!\n')
	else:
		data = image.to(DEVICE)
		logits, probas = model(data)
		predict_levels = probas > 0.5
		predicted_label = torch.sum(predict_levels, dim=1)
		print('Class probabilities:', probas)
		print('Predicted class label:', predicted_label.item())
		print('Predicted age:', predicted_label.item())
