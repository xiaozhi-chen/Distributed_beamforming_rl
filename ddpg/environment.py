import numpy as np
import pdb
from utils import normalize

class Environment:
	"""
	SINR foromulas
	(Environment works with numpy objects !)
	
	"""
	def __init__(self, nb_AP=10, nb_Users=5, transmission_power=5.01, seed=0):
		self.M = nb_AP                  # number of Access Points
		self.K = nb_Users				# number of Users
		self.P = transmission_power
		self.state_dim = self.K
		self.action_dim = self.M * self.K
		self.observation_shape = (self.state_dim,)
		self.action_space = (self.action_dim, )
		np.random.seed(seed)

		# shape(G) = M * K
		self.G = np.random.exponential(scale=1.0,size=(self.M, self.K)).astype('float32')
		self.G = self.G[:, np.argsort(self.G.sum(axis=0))]
		

		W = np.random.uniform(high=1,size=(self.M, self.K)).astype('float32')
		W = normalize(W)
		self.W = W
		
		
		
	
	def sinr(self):
		"""
		Calculates the sinr (state)
		
		"""
#		pdb.set_trace()
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
		W = np.random.uniform(low=0,high=1,size=(self.M, self.K)).astype('float32')
		W = W/np.linalg.norm(W, axis=1).reshape(self.M,1)
		self.W = W
	
		return self.sinr()


	def step(self, action_t):
		# action_t is of dimension K*1
#		self.W[agentId,:] = action_t
		self.W = action_t.reshape(self.M, self.K)

		norm = np.linalg.norm(self.W, axis=1).reshape(self.M,1)
		if 0.0 in norm:
			norm = norm + 1e-4

		self.W = self.W/norm
		state_t_pls_1 = self.sinr()
		rwd_t = np.sum(np.log2(1+state_t_pls_1))
		done_t = 0.0
		return state_t_pls_1, rwd_t, np.float32(done_t), self.W.reshape(1,-1)

if __name__ == "__main__":
	obj = Environment()
	print("test sinr() {}".format(obj.sinr()))