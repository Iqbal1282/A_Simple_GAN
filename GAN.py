import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal


import numpy as np
import json
import os
import pickle as pkl
import matplotlib.pyplot as plt



class GAN():
	def __init__(self):
		self.discriminator_input_dim = (28, 28, 1)
		self.weight_init = RandomNormal(mean = 0, stddev = 0.02)


		self.z_dim = 100
		self.generator_initial_dense_layer_size = (7, 7, 64)
		self.generator_batch_norm_momentum = 0.9 


		self.epoch = 0 
		self.d_losses = []
		self.g_losses = []


		self._discriminator_builder()
		self._generator_builder()
		self._build_adversarial()



	def _discriminator_builder(self):
		discriminator_input = Input(shape =self.discriminator_input_dim ,name = 'discriminator_input' )
		x = Conv2D(
			filters = 64,
			kernel_size= 5,
			strides = 2,
			padding = 'same',
			name = 'Conv2D_layer1',
			kernel_initializer = self.weight_init

			)(discriminator_input)

		x = Activation('relu')(x)
		x = Dropout(rate = 0.4)(x)

		# layer 2 
		x = Conv2D(
			filters = 64,
			kernel_size= 5,
			strides = 2,
			padding = 'same',
			name = 'Conv2D_layer2',
			kernel_initializer = self.weight_init

			)(x)

		x = Activation('relu')(x)
		x = Dropout(rate = 0.4)(x)

		# layer 3 

		x = Conv2D(
			filters = 128,
			kernel_size= 5,
			strides = 2,
			padding = 'same',
			name = 'Conv2D_layer3',
			kernel_initializer = self.weight_init

			)(x)

		x = Activation('relu')(x)
		x = Dropout(rate = 0.4)(x)

		# layer 4 

		x = Conv2D(
			filters = 128,
			kernel_size=5,
			strides = 2,
			padding = 'same',
			name = 'Conv2D_layer4',
			kernel_initializer = self.weight_init

			)(x)

		x = Activation('relu')(x)
		x = Dropout(rate = 0.4)(x)


		# layer 5
		x = Flatten()(x)
		discriminator_output = Dense(1, activation = 'sigmoid', kernel_initializer = self.weight_init)(x)


		self.discriminator_model = Model(discriminator_input, discriminator_output)


	def _generator_builder(self):

		generator_input = Input(shape = (self.z_dim,), name = 'generator_input')
		x = generator_input

		# layer1 

		x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer = self.weight_init)(x)
		x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
		x = Activation('relu')(x)

		x = Reshape(self.generator_initial_dense_layer_size)(x)

		# layer 

		x = UpSampling2D()(x)
		x = Conv2D(
					filters = 128
					, kernel_size = 5
					, padding = 'same'
					, name = 'layer2_upsample'
					, kernel_initializer = self.weight_init
				)(x)

		x = BatchNormalization(momentum = 0.9)(x)
		x = Activation('relu')(x)





		# layer 

		x = UpSampling2D()(x)
		x = Conv2D(
					filters = 64
					, kernel_size = 5
					, padding = 'same'
					, name = 'layer3_upsample'
					, kernel_initializer = self.weight_init
				)(x)

		x = BatchNormalization(momentum = 0.9)(x)
		x = Activation('relu')(x)


		# layer


		x = Conv2DTranspose(
					filters = 64
					, kernel_size = 5
					, padding = 'same'
					, strides = 1
					, name = 'layer_4'
					, kernel_initializer = self.weight_init
					)(x)

		x = BatchNormalization(momentum = 0.9)(x)
		x = Activation('relu')(x)


		# layer last layer  

		x = Conv2DTranspose(
					filters = 1
					, kernel_size = 5
					, padding = 'same'
					, strides = 1
					, name = 'generator_conv_last' 
					, kernel_initializer = self.weight_init
					)(x)


		x = Activation('tanh')(x)


		generator_output = x
		self.generator_model = Model(generator_input, generator_output)


	def set_trainable(self, m, val):
		m.trainable = val
		for l in m.layers:
			l.trainable = val


	def _build_adversarial(self):

		self.discriminator_model.compile(
				optimizer = RMSprop(lr =0.0008),
				loss = 'binary_crossentropy',
				metrics = ['accuracy']

			)

		self.set_trainable(self.discriminator_model, False)

		model_input = Input(shape = (self.z_dim,), name = 'model_input')
		model_output = self.discriminator_model(self.generator_model(model_input),)
		self.model = Model(model_input, model_output)

		self.model.compile(
			optimizer = RMSprop(lr = 0.0004),
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
			)

		self.set_trainable(self.discriminator_model, True)



	def train_discriminator(self, x_train, batch_size, using_generator):

		valid = np.ones((batch_size,1))
		fake = np.zeros((batch_size,1))

		if using_generator:
			true_imgs = next(x_train)[0]
			if true_imgs.shape[0] != batch_size:
				true_imgs = next(x_train)[0]
		else:
			idx = np.random.randint(0, x_train.shape[0], batch_size)
			true_imgs = x_train[idx]
		
		noise = np.random.normal(0, 1, (batch_size, self.z_dim))
		gen_imgs = self.generator_model.predict(noise)

		d_loss_real, d_acc_real =   self.discriminator_model.train_on_batch(true_imgs, valid)
		d_loss_fake, d_acc_fake =   self.discriminator_model.train_on_batch(gen_imgs, fake)
		d_loss =  0.5 * (d_loss_real + d_loss_fake)
		d_acc = 0.5 * (d_acc_real + d_acc_fake)

		return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

	def train_generator(self, batch_size):
		valid = np.ones((batch_size,1))
		noise = np.random.normal(0, 1, (batch_size, self.z_dim))
		return self.model.train_on_batch(noise, valid)


	def train(self, x_train, batch_size, epochs, run_folder
	, print_every_n_batches = 50
	, using_generator = False):
		temp_path = os.path.join(run_folder, 'weights')
		if not os.path.exists(temp_path):
			os.makedirs(temp_path)

		for epoch in range(self.epoch, self.epoch + epochs):

			d = self.train_discriminator(x_train, batch_size, using_generator)
			g = self.train_generator(batch_size)

			print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

			self.d_losses.append(d)
			self.g_losses.append(g)

			if epoch % print_every_n_batches == 0:
				self.sample_images(run_folder)
				self.model.save_weights(os.path.join(temp_path, 'weights-%d.h5' % (epoch)))
				self.model.save_weights(os.path.join(temp_path ,'weights.h5'))
				self.save_model(run_folder)

			self.epoch += 1





	def sample_images(self, run_folder):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.z_dim))
		gen_imgs = self.generator_model.predict(noise)

		gen_imgs = 0.5 * (gen_imgs + 1)
		gen_imgs = np.clip(gen_imgs, 0, 1)

		fig, axs = plt.subplots(r, c, figsize=(15,15))
		cnt = 0

		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
				axs[i,j].axis('off')
				cnt += 1
		temp_path = os.path.join(run_folder, "images")
		if not os.path.exists(temp_path):
			os.makedirs(temp_path)


		fig.savefig(os.path.join(temp_path, "sample_%d.png"% self.epoch) )
		plt.close()




	def save_model(self, run_folder):
		self.model.save(os.path.join(run_folder, 'model.h5'))
		self.discriminator_model.save(os.path.join(run_folder, 'discriminator.h5'))
		self.generator_model.save(os.path.join(run_folder, 'generator.h5'))
		#pkl.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

	def load_weights(self, filepath):
		self.model.load_weights(filepath)

# model = GAN()
# model.discriminator_model.summary()
# model.model.summary()




# import os 
# import numpy as np 
# import matplotlib.pyplot as plt 


# def load_safari(folder):

#     mypath = os.path.join("./data", folder)
#     txt_name_list = []
#     for (dirpath, dirnames, filenames) in os.walk(mypath):
#         for f in filenames:
#             if f != '.DS_Store':
#                 txt_name_list.append(f)
#                 break

#     slice_train = int(80000/len(txt_name_list))  ###Setting value to be 80000 for the final dataset
#     i = 0
#     seed = np.random.randint(1, 10e6)

#     for txt_name in txt_name_list:
#         txt_path = os.path.join(mypath,txt_name)
#         x = np.load(txt_path)
#         x = (x.astype('float32') - 127.5) / 127.5
#         # x = x.astype('float32') / 255.0
        
#         x = x.reshape(x.shape[0], 28, 28, 1)
        
#         y = [i] * len(x)  
#         np.random.seed(seed)
#         np.random.shuffle(x)
#         np.random.seed(seed)
#         np.random.shuffle(y)
#         x = x[:slice_train]
#         y = y[:slice_train]
#         if i != 0: 
#             xtotal = np.concatenate((x,xtotal), axis=0)
#             ytotal = np.concatenate((y,ytotal), axis=0)
#         else:
#             xtotal = x
#             ytotal = y
#         i += 1
        
#     return xtotal, ytotal



# (x_train, y_train) = load_safari('camel')

# gan = GAN()

# BATCH_SIZE = 64
# EPOCHS = 400
# PRINT_EVERY_N_BATCHES = 50


# gan.train(     
#     x_train
#     , batch_size = BATCH_SIZE
#     , epochs = EPOCHS
#     , run_folder = 'history/GAN/Camel_dataset'
#     , print_every_n_batches = PRINT_EVERY_N_BATCHES
# )


