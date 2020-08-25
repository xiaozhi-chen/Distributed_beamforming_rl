import pdb
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

class Actor():
	""" 
	Policy(State) = action

	NEURAL NETWORK MODEL:
	Input -> hidden-layer with 256 units (relu) -> hidden-layer with 128 units (sigmoid) -> output (sigmoid)
	
	Sigmoid is chosen because of the constraint:  0<=Wij<=1
	"""
	
	def __init__(self, state_dim, action_dim, lr, action_bound_range):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound_range = action_bound_range
		self.lr = lr
		self.model = self.build_model()
		self.target = self.build_model()
		self.adam_optimizer = Adam(learning_rate=lr)

	
	def build_model(self):
		state = Input(shape=self.state_dim)
		x = Dense(256, activation='relu')(state)
		x = Dense(128, activation='relu')(x)

		out = Dense(self.action_dim, activation='sigmoid')(x)

		#ensure np.linalg.norm(out) <= 1
		out = Lambda(lambda i: tf.math.l2_normalize(i))(out)

		return Model(inputs=state, outputs=out)




if __name__ == '__main__':
	test = Actor(5, 5, 0.001, 1)
	pdb.set_trace()