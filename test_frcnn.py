from __future__ import division
import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model


from keras_frcnn import roi_helpers
import keras_frcnn.vgg as nn

def format_img_size(img):
	""" formats the image size based on config """
	img_min_side = float(600)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img_channel_mean = [103.939, 116.779, 123.68]
	img[:, :, 0] -= img_channel_mean[0]
	img[:, :, 1] -= img_channel_mean[1]
	img[:, :, 2] -= img_channel_mean[2]
	img /= 1.0
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img):
	img, ratio = format_img_size(img)
	img = format_img_channels(img)
	return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


################################################################################config

class_mapping = {'pistol': 0, 'bg': 1}
class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
num_rois = 32
num_features = 512
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn.nn_base(img_input, trainable=True)
anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [0.7071067811865475, 1.414213562373095], [1.414213562373095, 0.7071067811865475]]

classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
rpn_stride = 16

####################################################################################

# define the RPN, built on the base layers
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)


print('Loading weights')
model_rpn.load_weights('./models/model_frcnn.hdf5', by_name=True)
model_classifier.load_weights('./models/model_frcnn.hdf5', by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5

visualise = True


###########################################


def detect(frame):
	img = frame

	X, ratio = format_img(img)
	X = np.transpose(X, (0, 2, 3, 1))
	[Y1, Y2, F] = model_rpn.predict(X)
	R = roi_helpers.rpn_to_roi(Y1, Y2, K.image_data_format(), overlap_thresh=0.7)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]
	bboxes = {}
	probs = {}
	for jk in range(R.shape[0]//num_rois + 1):
		ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break
		if jk == R.shape[0]//num_rois:
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded
		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
		for ii in range(P_cls.shape[1]):
			if (np.max(P_cls[0, ii, :]) < bbox_threshold) or (np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1)):
				continue
			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
			print(cls_name)
			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])

			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= classifier_regr_std[0]
				ty /= classifier_regr_std[1]
				tw /= classifier_regr_std[2]
				th /= classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([rpn_stride*x, rpn_stride*y, rpn_stride*(x+w), rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.1)

		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)


	print(all_dets)
	cv2.imshow('img', img)
	cv2.waitKey(0)

#####################################


filepath = './dataset/WeaponS/armas (458).jpg'
frame = cv2.imread(filepath)
detect(frame)

######################################


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if ret:
#         detect(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()