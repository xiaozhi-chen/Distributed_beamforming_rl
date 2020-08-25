from .env_wrapper import EnvWrapper
import numpy as np
import torch
from torch.distributions.exponential import Exponential
import pdb


class Beamforming(EnvWrapper):
    def __init__(self, config):
        self.config = config
        torch.manual_seed(config["random_seed"])

        self.M = int(config["M"])
        self.K = int(config["K"])

        self.G = Exponential(1).sample((self.M, self.K)).float().cpu().detach().numpy()
        self.G = self.G[:, np.argsort(self.G.sum(axis=0))]
        self.P = float(config["transmission_power"])


    def sinr(self, W):
        W2 = np.square(W)
        gamma = np.zeros(self.K, dtype='float32')

        for k in range(self.K):
            nom = np.dot(self.G[:,k].reshape(1,self.M), W2[:,k].reshape(self.M,1))

#            denom = (self.G.sum(axis=1) - self.G[:,k]).reshape(self.M, 1)
#            denom += self.G[:, list(range(k))].sum(axis=1).reshape(self.M, 1)
#            denom += (self.G.sum(axis=1) * (self.K-2) + self.G[:,k]).reshape(self.M, 1)

            denom = self.G[:, list(range(k))].sum(axis=1)
            denom = denom.reshape(self.M, 1)
            denom = np.dot(W2[:,k].reshape(1,self.M), denom)
            denom += np.float32(1.0/self.P)

            gamma[k] = nom/denom

        return torch.from_numpy(gamma).float().to("cpu")

    def reset(self):
        W = torch.rand(self.M, self.K)
        W = torch.nn.functional.normalize(W, p=2)
        return self.sinr(W)

    def step(self, action):
        state = self.sinr(action.reshape(self.M, self.K))
        reward = (1+state).log2().sum()
        return state, reward, 0


