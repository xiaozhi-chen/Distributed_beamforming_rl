import pdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environment import Environment
from ddpg import DDPG
import gym


if __name__=="__main__":
	nb_AP = 15
	nb_Users = 5
	p= np.float32(10**0.7)
	timestamp = 10000
#	env = gym.make("Pendulum-v0")

	env = Environment(nb_AP= nb_AP, nb_Users=nb_Users, transmission_power=p, seed=0)
	big_boss = DDPG(env, timestamp=timestamp, actor_weights=None, critic_weights=None)
	actor_weights, critic_weights = big_boss.train()


"""	for agentId in range(1,2):
		env = Environment(agentId = agentId, nb_AP = nb_AP, nb_Users = nb_Users)
		agent = DDPG(env, discount_factor=0)
		agent.train()
"""