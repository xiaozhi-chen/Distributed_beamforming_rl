import numpy as np
import pdb

from buffer import Replay_Buffer
from utils import normalize

class Environment:
	"""
	SINR foromulas
	(Environment works with numpy objects !)
	
	"""
	def __init__(self, nb_AP=10, nb_Users=5, transmission_power=5.01,
				max_buffer_size=1000000,
				batch_size=200):
		self.M = nb_AP                  # number of Access Points
		self.K = nb_Users				# number of Users
		self.P = transmission_power
		self.state_dim = self.K
		self.action_dim = self.K
		self.observation_shape = (self.state_dim,)
		self.action_space = (self.action_dim, )
		np.random.seed(10)

		# shape(G) = M * K
		G = np.random.exponential(scale=1.0,size=(self.M, self.K)).astype('float32')
				
		W = np.random.uniform(high=1,size=(self.M, self.K)).astype('float32')
		W = normalize(W)

		self.W = W
		self.G = G
	

		#shared buffer
		self.max_buffer_size = max_buffer_size
		self.batch_size = batch_size



	def sinr(self):
		"""
		Calculates the sinr (state)
		
		"""
		
		W2 = np.square(self.W)

		gamma = np.zeros(self.K, dtype='float32')

		for k in range(self.K):
			nom = np.dot(self.G[:,k].reshape(1,self.M), W2[:,k].reshape(self.M,1))
			
#			denom = 2.0*(self.G.sum(axis=1) - self.G[:,k])
#			denom += self.G.sum(axis=1) * (self.K-2) + self.G[:,k]
			
			denom = self.G[:, list(range(k))].sum(axis=1)
			denom = denom.reshape(self.M, 1)
			denom = np.dot(W2[:,k].reshape(1,self.M), denom)
			denom += np.float32(1.0/self.P)

			gamma[k] = nom/denom

		return gamma.astype(np.float32)
	
	def reset(self):
		self.G = np.random.exponential(scale=1.0,size=(self.M, self.K)).astype('float32')
		self.G = self.G[:, np.argsort(self.G.sum(axis=0))]
				
		self.W = np.random.uniform(high=1,size=(self.M, self.K)).astype('float32')
		self.W = normalize(self.W)
	
		return self.sinr()

	def step(self, action_t, agentId):
		# action_t is of dimension K*1
		self.W[agentId,:] = action_t
		assert (np.linalg.norm(self.W,axis=1) <= (np.ones(self.M)+1e-4)).sum() == self.M
#		if agentId == 3 or agentId==4:
#			print(self.W, agentId)

		state_t_pls_1 = self.sinr()
		rwd_t = np.sum(np.log2(1+state_t_pls_1))
		done_t = 0.0
		return state_t_pls_1, rwd_t, np.float32(done_t), self.W[agentId]
	
	def set_W(self, W):
		self.W = W

if __name__ == "__main__":
	obj = Environment()
	print("test sinr() {}\ntest step(action, id) {}\ntest reset() {}".format(obj.sinr(), obj.step(np.random.randn(obj.action_dim), 1), obj.reset()))
