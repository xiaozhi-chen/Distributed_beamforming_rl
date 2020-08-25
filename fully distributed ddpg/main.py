import multiprocessing as mp
import numpy as np
from environment import Environment
from ddpg import DDPG
import datetime
from tensorflow import summary




def worker(env, agentId, timestamp, logger_folder, action_queue, matrix_queue, max_epsiode_steps):
	agent = DDPG(env, agentId, logger_folder, timestamp=timestamp, max_epsiode_steps=max_epsiode_steps)
	agent.train(action_queue, matrix_queue)


def boss(env, nb_AP, nb_Users, action_queues, matrix_queues, logger_folder, max_epsiode_steps):
	step = 0
	writer = summary.create_file_writer('logs/'+logger_folder+'/boss')
	writer.set_as_default()
	summary.experimental.set_step(step)
	
	while True:
		step += max_epsiode_steps
		W = np.zeros((nb_AP, nb_Users)).astype('float32')
		for i in range(nb_AP):
			W[i:] = action_queues[i].get()
		
		W = W/np.linalg.norm(W,axis=1).reshape(W.shape[0],1)
		for q in matrix_queues:
			q.put(W)

		env.set_W(W)
		r = np.sum(np.log2(1+env.sinr()))
		summary.scalar(name='Episode/Reward', data=r, step=step)
		print("********* \nReward {0:5.6f} Step {1: 6} norm {2: 4.5f}\n************".format(np.sum(np.log2(1+env.sinr())), step, np.linalg.norm(W)))

if __name__=="__main__":
	# parameters
	nb_AP = 15
	nb_Users = 5
	p = 5.01
	timestamp = 10000
	max_epsiode_steps = 1000
	
	#Tensorboard logger
	logger_folder = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

	action_queues = [mp.Queue() for i in range(nb_AP)]
	matrix_queues = [mp.Queue() for i in range(nb_AP)]


	env = Environment(nb_AP= nb_AP, nb_Users=nb_Users, transmission_power=p)
	processes = []
	
	for agentId in range(nb_AP):
		pros = mp.Process(target=worker, args=(env, agentId, timestamp, logger_folder, 
							action_queues[agentId], matrix_queues[agentId], max_epsiode_steps), name='agent-'+str(agentId))
		processes.append(pros)

	pros = mp.Process(target=boss, args=(env, nb_AP, nb_Users, action_queues, matrix_queues, logger_folder, max_epsiode_steps), name='boss')
	processes.append(pros)
	
	for pros in processes:
		pros.start()

	for pros in processes:
		pros.join()
