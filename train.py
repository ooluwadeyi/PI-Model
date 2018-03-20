from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import config, warnings
import scipy
from collections import OrderedDict 
import json, math
import pickle
import os
import sys
import scipy 
import skimage.io as io
import os
import re
import sys
import tarfile
from six.moves import urllib
import config


'''
Put the calculation of the cross entropy error in a if statement and check if the data has  label that is not -1, then calculate the cross entropy error otherwise,
make the cross entropy error zero.
For the dataset, set the labels of the unlabeled samples to -1 so that you can use it to determine if the sample is labeled or not

'''


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('log_dir', 'tmp/logs',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, 
                            """Train the model using fp16.""")



IMAGE_SIZE = 32
NUM_CLASSES = 11

def save_image(filename, img):
    if len(img.shape) == 3:
        if img.shape[0] == 1:            
            img = img[0] # CHW -> HW (saves as grayscale)
        else:            
            img = np.transpose(img, (1, 2, 0)) # CHW -> HWC (as expected by toimage)


    #scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(filename)
    #io.imsave(filename, img)

def weights_func(val):

	return val



def make_unlabeled_samples(X, y, factor):

	size = X.shape[0]
	unlabeled_size = int(size*factor)
	rand = np.random.permutation(size)
	if unlabeled_size == size:
		uniq_index = rand[:unlabeled_size-1]
	else:
		uniq_index = rand[:unlabeled_size]

	for i in uniq_index:
		y[i] = 10

	return X, y



def load_cifar_10():
   
	def load_cifar_batches(filenames):
		if isinstance(filenames, str):
			filenames = [filenames]
		images = []
		labels = []
		for fn in filenames:
			with open(os.path.join(config.data_dir, fn), 'rb') as f:
				data = pickle.load(f, encoding='bytes')
			images.append(np.asarray(data[b'data'], dtype='float32').reshape(-1, 3, 32, 32) / np.float32(255))
			labels.append(np.asarray(data[b'labels'], dtype='int32'))
		return np.concatenate(images), np.concatenate(labels)

	X_train, y_train = load_cifar_batches(['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
	X_test, y_test = load_cifar_batches('test_batch')

	X_train = X_train.transpose([0, 2, 3, 1])
	X_test = X_test.transpose([0, 2, 3, 1])

	return X_train, y_train, X_test, y_test


def unpickle(file):
    
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def whiten_norm(x):

	x = x - np.mean(x, keepdims=True)
	x = x / (np.mean(x ** 2, keepdims=True) ** 0.5)

	x = np.reshape(x, [-1, 32, 32, 3])
	return x

	

def prepare_dataset (X_train, y_train, X_test, y_test, num_classes):  #(X_train, y_train, X_test, y_test, num_classes)

	'''This function performs the translation augmentation of the input images  by amount set in config.augment_translation and also mask the labels for some input data to be treated as unlabeled'''

	if config.whiten_inputs == 'norm':
		X_train = whiten_norm(X_train)
		X_test = whiten_norm(X_test)
	
	
	
	p = config.augment_translation 
	if p > 0:
		X_train = np.pad(X_train, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')
		X_test = np.pad(X_test, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

    # Random shuffle.
	
	indices = np.arange(len(X_train))
	np.random.shuffle(indices)
	X_train = X_train[indices]
	y_train = y_train[indices]    

    # Construct mask_train. It has a zero where label is unknown, and one where label is known.
	if config.num_labels == 'all':
        # All labels are used.
		mask_train = np.ones(len(y_train), dtype=np.float32)
		print("Keeping all labels.")
	else:
		#Assign labels to a subset of inputs.
		num_img = min(num_classes, 20)
		max_count = config.num_labels // num_classes   
		print("Keeping %d labels per class." % max_count)
		img_count = min(max_count, 32)
		#print(X_train.shape[1])  # I expect 32
		label_image = np.zeros((32 * num_img, 32 * img_count, 3))
		#abel_image = np.zeros((36,36,3)) 
	
		mask_train = np.zeros(len(y_train), dtype=np.float32)
	
		count = [0] * num_classes # =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		for i in range(len(y_train)):  # i in [ 0, 1, 2, 3, .... 49,999]
			label = y_train[i]  # label is between (0 and 9)
			if count[label] < max_count:  #max_count = 50
				mask_train[i] = 1.0
				if count[label] < img_count and label < num_img:
					label_image[label * 32 : (label + 1) * 32, count[label] * 32 : (count[label] + 1) * 32, :] = X_train[i, p:p+32, p:p+32, :]
				count[label] += 1
		


    # Zero out masked-out labels for maximum paranoia.
	# for i in range(len(y_train)):
	# 	if mask_train[i] != 1.0:
	# 		y_train[i] = 0


	return X_train, y_train, mask_train, X_test, y_test


###################################################################################################
# Training utils.
###################################################################################################



def rampup(epoch):
    if epoch < config.rampup_length:
        p = max(0.0, float(epoch)) / float(config.rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0




def rampdown(epoch):
    if epoch >= (config.num_epochs - config.rampdown_length):
        ep = (epoch - (config.num_epochs - config.rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / config.rampdown_length)
    else:
        return 1.0




###################################################################################################
# Training iterators.
###################################################################################################

def iterate_minibatches(inputs, targets, batch_size):

	''' This function constructs the minibatches for testing '''


	inputs = inputs.astype(np.float32)
	assert len(inputs) == len(targets)
	num = len(inputs)

	indices = np.arange(num) 
	crop = config.augment_translation
	for start_idx in range(0, num, batch_size):
		if start_idx + batch_size <= num:
			excerpt = indices[start_idx : start_idx + batch_size]
			yield (len(excerpt), inputs[excerpt, crop:crop+32, crop:crop+32, :], targets[excerpt]) # yield

def iterate_minibatches_augment_pi(inputs, labels, mask, batch_size):

	''' This function constructs the minibatches for training '''
	
	assert len(inputs) == len(labels) == len(mask)
	crop = config.augment_translation

	num = len(inputs)
	
	if config.max_unlabeled_per_epoch is None:
		indices = np.arange(num)
	else:        
		labeled_indices   = [i for i in range(num) if mask[i] > 0.0]
		

		unlabeled_indices = [i for i in range(num) if mask[i] == 0.0]
	
		np.random.shuffle(unlabeled_indices)
		indices = labeled_indices + unlabeled_indices[:config.max_unlabeled_per_epoch] # Limit the number of unlabeled inputs per epoch.
		
		indices = np.asarray(indices)
		num = len(indices)

	np.random.shuffle(indices)
	
	for start_idx in range(0, num, batch_size):
		
		if start_idx + batch_size <= num:
			excerpt = indices[start_idx : start_idx + batch_size]
			noisy_a, noisy_b = [], []
			for img in inputs[excerpt]:
				if config.augment_mirror and np.random.uniform() > 0.5:
					img = img[:, :, ::-1]

				t = config.augment_translation
				ofs0 = np.random.randint(-t, t + 1) + crop
				ofs1 = np.random.randint(-t, t + 1) + crop
				img_a = img[ofs0:ofs0+32, ofs1:ofs1+32, :]
				ofs0 = np.random.randint(-t, t + 1) + crop
				ofs1 = np.random.randint(-t, t + 1) + crop
				img_b = img[ofs0:ofs0+32, ofs1:ofs1+32, :]
				noisy_a.append(img_a)
				noisy_b.append(img_b)
				

			yield (len(excerpt), excerpt, noisy_a, noisy_b, labels[excerpt])	# yield mask[excerpt]





###################################################################################################
# Network construction.
###################################################################################################
# def _activation_summary(x):
# 	"""Helper to create summaries for activations.

# 	Creates a summary that provides a histogram of activations.
# 	Creates a summary that measures the sparsity of activations.

# 	Args:
# 	x: Tensor
# 	Returns:
# 	nothing
# 	"""
# 	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
# 	# session. This helps the clarity of presentation on tensorboard.
# 	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
# 	tf.summary.histogram(tensor_name + '/activations', x)
# 	tf.summary.scalar(tensor_name + '/sparsity',
# 	                                   tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable

	Returns:
	Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
	name: name of the variable
	shape: list of ints
	stddev: standard deviation of a truncated Gaussian
	wd: add L2Loss weight decay multiplied by this float. If None, weight
	    decay is not added for this Variable.

	Returns:
	Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
	  name,
	  shape,
	  tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var






def build_network(images, keep_prob_1, keep_prob_2):# num_input_channels, num_classes): images

	"""Build the CIFAR-10 according to the architecture provided in the paper """

	# keep_prob_1 = tf.placeholder(tf.float32)
	# keep_prob_2 = tf.placeholder(tf.float32)
	weights_decay = 0.0004

	with tf.variable_scope('Network', reuse=tf.AUTO_REUSE) as scope:

		with tf.variable_scope('conv1a') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 128], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
			biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1a = tf.nn.leaky_relu(pre_activation, alpha=0.1,  name=scope.name)
		  #_activation_summary(conv1)

		with tf.variable_scope('conv1b') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(conv1a, kernel, [1, 1, 1, 1], padding='SAME')
			biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1b = tf.nn.leaky_relu(pre_activation, alpha=0.1,  name=scope.name)
		  #_activation_summary(conv1)

		with tf.variable_scope('conv1c') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(conv1b, kernel, [1, 1, 1, 1], padding='SAME')
			biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1c = tf.nn.leaky_relu(pre_activation, alpha=0.1,  name=scope.name)
		  #_activation_summary(conv1)	


		# pool1
		pool1 = tf.nn.max_pool(conv1c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

		dropout1 = tf.nn.dropout(pool1, keep_prob_1)
		# norm1
		#norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,  name='norm1')

		# conv2 shape is 16 x 16 here
		with tf.variable_scope('conv2a') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(dropout1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv2a = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)
		  #_activation_summary(conv2)

		with tf.variable_scope('conv2b') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(conv2a, kernel, [1, 1, 1, 1], padding='SAME')
			biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv2b = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)
		  #_activation_summary(conv2)  

		with tf.variable_scope('conv2c') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(conv2b, kernel, [1, 1, 1, 1], padding='SAME')
			biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv2c = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)
		  #_activation_summary(conv2)  

		# norm2
		#norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
		# pool2
		pool2 = tf.nn.max_pool(conv2c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

		dropout2 = tf.nn.dropout(pool2, keep_prob_2)

		#conv 3a  shape is 8 X 8 here
		with tf.variable_scope('conv3a') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(dropout2, kernel, [1, 1, 1, 1], padding='VALID')
			biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv3a = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)
		  #_activation_summary(conv2)  

		with tf.variable_scope('conv3b') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 256], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(conv3a, kernel, [1, 1, 1, 1], padding='VALID')
			biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv3b = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)

		with tf.variable_scope('conv3c') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 128], stddev=5e-2, wd=weights_decay)
			conv = tf.nn.conv2d(conv3b, kernel, [1, 1, 1, 1], padding='VALID')
			biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv3c = tf.nn.leaky_relu(pre_activation, alpha=0.1, name=scope.name)	

		pool3 = tf.nn.avg_pool(conv3c, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME', name='pool2')	



		with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
			reshape = tf.reshape(pool3, [config.minibatch_size, -1])
			dim = reshape.get_shape()[1].value
			weights = _variable_with_weight_decay('weights', shape=[dim, 10],  stddev=0.04, wd=weights_decay)
			biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.1))
			local3 = tf.matmul(reshape, weights) + biases
			#_activation_summary(local3)

	

	return local3#tf.nn.softmax(local3)



		
	

	
	
	# with tf.variable_scope("Network", reuse= tf.AUTO_REUSE) as scope:

	# 	noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=1.0, dtype=tf.float32, name="noise")
	# 	input_layer = noise + tf.reshape(input_layer, [-1, 32, 32, 3])   
		
	# 	conv1a = tf.layers.conv2d(inputs=input_layer, filters=128, kernel_size=[3, 3], padding="same",	activation=tf.nn.leaky_relu, name="conv1a")		
	# 	conv1b = tf.layers.conv2d(inputs=conv1a, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, name="conv1b")
	# 	conv1c = tf.layers.conv2d(inputs=conv1b, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, name="conv1c")   
	# 	pool1 = tf.layers.max_pooling2d(inputs=conv1c, pool_size=[2, 2], strides=1, name="pool1")
	# 	dropout1 = tf.layers.dropout(inputs=pool1, rate=0.5, training=True, name="dropout1")
	# 	conv2a = tf.layers.conv2d(inputs=dropout1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, name="conv2a")
	# 	conv2b = tf.layers.conv2d(inputs=conv2a, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, name="conv2b")
	# 	conv2c = tf.layers.conv2d(inputs=conv2b, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, name="conv2c")
	# 	pool2 = tf.layers.max_pooling2d(inputs=conv2c, pool_size=[2, 2], strides=1, name="pool2")
	# 	dropout2 = tf.layers.dropout(inputs=pool2, rate=0.5, training=True, name="dropout2")
	# 	conv3a = tf.layers.conv2d(inputs=dropout2, filters=512, kernel_size=[3, 3], padding="valid", activation=tf.nn.leaky_relu, name="con3a") 
	# 	conv3b = tf.layers.conv2d(inputs=conv3a, filters=256, kernel_size=[1, 1], padding="valid", activation=tf.nn.leaky_relu, name="conv3b")
	# 	conv3c = tf.layers.conv2d(inputs=conv3a, filters=128, kernel_size=[1, 1], padding="valid", activation=tf.nn.leaky_relu, name="conv3c")
					
	# 	pool3 = tf.reduce_mean(conv3c, [1,2], name="pool3")
	# 	dense = tf.layers.dense(inputs=pool3, units=10, activation=tf.nn.softmax, name="dense")

	

	# return dense

###################################################################################################
# Convert labels to one-hot
###################################################################################################


def dense_to_one_hot(data, num_classes=10):

	"""Convert class labels from scalars to one-hot vectors"""
	
	integer_encoded = data
	
	# one hot encode
	onehot_encoded = list()
	for value in integer_encoded:
		letter = [0 for _ in range(num_classes)]
		if value == 10:
			onehot_encoded.append(letter)
			continue

		letter[int(value)] = 1
		onehot_encoded.append(letter)
	
	
	
	return onehot_encoded


###################################################################################################
# Main training function.
###################################################################################################


def run_training():
	
	

	# Load the dataset.
	print("Loading dataset '%s'..." % config.dataset)
	X_train, y_train, X_test, y_test = load_cifar_10()

	X_train, y_train = make_unlabeled_samples(X_train, y_train, config.unlabeled_factor)
	#X_test, y_test = make_unlabeled_samples(X_train, y_train, config.test_unlabeled_factor)
	#print(y_train[:50])

	#X_test,y_test = make_unlabeled_samples(X_train, y_test, config.unlabeled_factor)
	
	





	
	#save_image('train-image1.png', X_train[0])
	# save_image('train-image2.png', X_train[1])
	# save_image('train-image25.png', X_train[25])		
	num_classes = 10 # len(set(y_train))

	#num_classes = config.num_classes

	#assert(set(y_train) == set(y_test) == set(range(num_classes))) # Check that all labels are in range [0, num_classes-1]
	print("Found %d classes in training set, %d in test set." % (len(set(y_train)), len(set(y_test))))

	# Prepare dataset and print stats.

	X_train, y_train, mask_train, X_test, y_test = prepare_dataset(X_train, y_train, X_test, y_test, num_classes) #X_test, y_test

	print("Got %d training inputs, out of which %d are labeled." % (len(X_train), sum(mask_train)))
	print("Got %d test inputs." % len(X_test))
	#save_image('Prep_train-image25.png', X_train[25])

	print("Network type is '%s'." % config.network_type)

	#n, indices, inputs_a, inputs_b, labels = next(iterate_minibatches_augment_pi(X_train, y_train, mask_train, config.minibatch_size))

	

	
	# labels_ = dense_to_one_hot(labels)
	# labels_ = np.asfarray(labels_)

	# for i in range(config.minibatch_size):
	# 	print(np.argmax(labels_[i], axis= -1))




	
	#print(inputs_a[0])

	#inputs_aa = inputs_a[15]#/np.amax(inputs_a[10])
	#inputs_bb = inputs_b[15]#/np.amax(inputs_b[10])
	#print(inputs_bb)
	#print("label is : ", labels[15])
	#save_image( 'noisy_input1.png', inputs_aa) # /np.float32(255))
	#save_image('noisy_input2.png', inputs_bb)
	
	scaled_unsup_weight_max = config.unsup_weight_max
	unsup_weight = 0.0

	graph = tf.Graph()

	x_train = tf.placeholder(tf.float32, shape=[config.minibatch_size, 32, 32, 3], name= 'input1')
	x_train_b = tf.placeholder(tf.float32, shape=[config.minibatch_size, 32, 32, 3], name= 'input2')
	y_ = tf.placeholder(tf.float32, shape = [config.minibatch_size, 10], name ='label')

	keep_prob_1 = tf.placeholder(tf.float32)
	keep_prob_2 = tf.placeholder(tf.float32)


	learning_rate = config.learning_rate_max
	adam_beta1 = config.adam_beta1
	adam_beta2 = config.adam_beta2
	adam_epsilon = config.adam_epsilon


	train_prediction = build_network(x_train, keep_prob_1, keep_prob_2) # The first branch
	train_prediction_b = build_network(x_train_b, keep_prob_1, keep_prob_2) # The second branch
	

	# Training loss.
	#sentinel2 = tf.Variable(-1, name='sentinel')

	#sentinel2 = tf.greater(tf.constant(10, dtype='int64'), tf.argmax(y_[0], axis= -1))

	weights = list()

	#weights = [1.0 for i in range(config.minibatch_size) if ]


	for i in range(config.minibatch_size):

		sentinel = tf.equal(tf.constant(0, dtype='int64'), tf.argmax(y_[i], axis= -1))

		val = tf.cond(sentinel,  lambda: weights_func(1.0), lambda: weights_func(0.0))	# reverse it back 	
		weights.append(val)

	weights = tf.convert_to_tensor(weights)
			
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.losses.softmax_cross_entropy(y_, train_prediction, weights=weights)
	#cross_entropy = tf.cond(tf.greater(tf.constant(10, dtype='int64'), tf.argmax(y_, axis=1)[0]), lambda: tf.constant(0.0), lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=train_prediction)), name ='cross_entropy')
	
	with tf.name_scope('train_loss'):

		train_loss = cross_entropy

		mean_sq_loss = tf.losses.mean_squared_error(train_prediction, train_prediction_b)
		train_loss += unsup_weight * mean_sq_loss
	#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(train_loss)
	with tf.name_scope('train'):

		train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=config.adam_epsilon).minimize(train_loss)
		#train_op = tf.train.AdamOptimizer(config.adam_epsilon).minimize(train_loss)
	#summary_op = tf.summary.merge_all()
	with tf.name_scope('Accuracy'):
		correct_train_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(y_, 1))
		train_accuracy = tf.reduce_mean(tf.cast(correct_train_prediction, tf.float32))*100


	tf.summary.scalar("cross_entropy_loss", cross_entropy)
	tf.summary.scalar("MSE", mean_sq_loss)
	tf.summary.scalar("Training Loss", train_loss)
	tf.summary.scalar("Accuracy", train_accuracy)

	summary_op = tf.summary.merge_all()

	
	#writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
	init = tf.global_variables_initializer()
	
	sess = tf.Session()
	sess.run(init)

	summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
	

	for epoch in range(config.start_epoch, config.num_epochs):

		
		
		rampup_value = rampup(epoch)
		rampdown_value = rampdown(epoch)

        
		
		# if epoch == 0:
		# 	if config.network_type == 'pi':
				
		# 		minibatches = next(iterate_minibatches_augment_pi(X_train, y_train, config.minibatch_size))#next(iterate_minibatches_augment_pi(X_train, np.zeros((len(X_train),)), np.zeros((len(X_train),)), config.minibatch_size)) 
							
		# 		n, indices, inputs_a, inputs_b, labels = minibatches #, mask

		# 		labels_ = dense_to_one_hot(labels)

		# 		sess.run(train_op, feed_dict={x_train: inputs_a, x_train_b: inputs_b, y_: labels_, keep_prob_1:0.5, keep_prob_2:0.5})
			
		learning_rate = rampup_value * rampdown_value * config.learning_rate_max
		adam_beta1 = rampdown_value * config.adam_beta1 + (1.0 - rampdown_value) * config.rampdown_beta1_target
		unsup_weight = rampup_value * scaled_unsup_weight_max
		
		n, indices, inputs_a, inputs_b, labels = next(iterate_minibatches_augment_pi(X_train, y_train, mask_train, config.minibatch_size))		
		#minibatches = next(iterate_minibatches_augment_pi(X_train, y_train,  config.minibatch_size))	#mask_train,
		#n, indices, inputs_a, inputs_b, labels = minibatches	#, mask 
		labels_ = dense_to_one_hot(labels)
		labels_ = np.asfarray(labels_)
		inputs_a = np.asfarray(inputs_a)
		inputs_b = np.asfarray(inputs_b)

		_, summary = sess.run([train_op, summary_op], feed_dict={x_train: inputs_a, x_train_b: inputs_b, y_:labels_, keep_prob_1:0.5, keep_prob_2:0.5})
		summary_writer.add_summary(summary, epoch)
		#print("Sentinel....:", sess.run(sentinel2, feed_dict={y_:labels_}))

		#print("Argamx.....:", sess.run(tf.argmax(y_,)[0], feed_dict={y_:labels_}))
		#print("Label..:", sess.run(y_, feed_dict={y_:labels_}))
		print("\n")

		if (epoch == 0 or epoch%2 == 0):

			print("Epoch: {}".format(epoch))
			# print("labels  \n", labels)
			# print("mask.. ", mask_train.shape)
			# print("Loss:...") 
			
			print(" Train Loss....:", sess.run(train_loss, feed_dict={x_train: inputs_a, x_train_b:inputs_b, y_:labels_, keep_prob_1:0.5, keep_prob_2:0.5}))
			print( "MSE.....:", sess.run(mean_sq_loss, feed_dict={x_train: inputs_a, x_train_b: inputs_b, keep_prob_1:0.5, keep_prob_2:0.5}))
			# print("learning_rate:....", learning_rate)
			# print("Adam Beta1:......", adam_beta1)
			print("Training Accuracy:....")
			# correct_train_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(labels_, 1))
			# train_accuracy = tf.reduce_mean(tf.cast(correct_train_prediction, tf.float32))*100
			print(sess.run(train_accuracy, feed_dict={x_train: inputs_a, x_train_b:inputs_b, y_:labels_, keep_prob_1:0.5, keep_prob_2:0.5}))
			#print("Sentinel....:", sess.run(sentinel, feed_dict={y_:labels_}))
			print("Weights....:", sess.run(weights, feed_dict={y_:labels_}))
			print("Unsup weight..:", unsup_weight)
			print("...\n")	
			


           
	# last epoch 
			
	print("Train Loss:...") 
	print(sess.run(train_loss, feed_dict={x_train: inputs_a, x_train_b: inputs_b, y_:labels_, keep_prob_1:0.5, keep_prob_2:0.5}))
	#print("learning_rate:....", learning_rate)
	#print("Adam Beta1:......", adam_beta1)
	print("Training Accuracy:....")
	# correct_train_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(labels_, 1))
	# train_accuracy = tf.reduce_mean(tf.cast(correct_train_prediction, tf.float32))*100
	print(sess.run(train_accuracy, feed_dict={x_train: inputs_a, x_train_b:inputs_b, y_:labels_, keep_prob_1:0.5, keep_prob_2:0.5}))
	#summary_writer.add_summary(summaries, epoch)



	#Test pass
	minibatches = next(iterate_minibatches(X_test, y_test, config.minibatch_size))	
	n, inputs, labels = minibatches
	labels_ = dense_to_one_hot(labels)
	labels_ = np.asfarray(labels_)
	inputs_a = np.asfarray(inputs_a)
	
	test_pred = build_network(inputs, keep_prob_1, keep_prob_2)	
	correct_prediction = tf.equal(tf.argmax(test_pred, 1), tf.argmax(labels_, 1))	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Test Accuarcy: .....\n")
	print(sess.run(accuracy*100, feed_dict={x_train: inputs_a, x_train_b:inputs_b, y_:labels_, keep_prob_1: 1.0, keep_prob_2: 1.0}))
        

       

     



if __name__ == '__main__':

	print("Starting up...")	
	run_training()
	print("Exiting...")



