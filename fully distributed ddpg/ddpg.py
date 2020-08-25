import numpy as np
import tensorflow as tf
import datetime
import pdb
import gym
import random
import time

from collections import deque
from tensorflow.keras.losses import MeanSquaredError
from actor import Actor
from critic import Critic
from utils import huber_loss, update_target_variables, normalize
from buffer import Replay_Buffer



class DDPG:
	def __init__(self, env, agentId, logger_folder,
				max_action=1,
				max_buffer_size=1000000,
				batch_size=200,
				tow = 0.005,
				discount_factor = 0.99,
				actor_learning_rate = 0.001,
				critic_learning_rate = 0.001,
				dtype='float32',
				timestamp = 10000,
				max_epsiode_steps = 1,
				n_warmup = 200,
				sigma=0.1):

		self.logger_folder = logger_folder
		self.max_buffer_size = max_buffer_size
		self.batch_size = batch_size
		self.max_epsiode_steps = max_epsiode_steps
		self.timestamp = timestamp

		self.n_warmup = n_warmup

		# 80 of time is for exploration
		self.dflt_dtype = dtype
		

		self.sigma = sigma
		self.tow = tow		
		self.gamma = discount_factor
		self.max_action = max_action
		np.random.seed(0)
		random.seed(0)
		
		self.env = env
		self.agentId = agentId

		if isinstance(self.env, gym.Env):
			self.state_dim = env.observation_space.shape[0] if env.observation_space.shape != tuple() else 1
			self.action_dim = env.action_space.shape[0] if env.action_space.shape != tuple() else 1
		else:
			self.state_dim = env.state_dim
			self.action_dim = env.action_dim

		self.actor = Actor(self.state_dim, self.action_dim, actor_learning_rate, max_action)

		self.critic = Critic(self.state_dim, self.action_dim, critic_learning_rate)

		update_target_variables(
			self.actor.target.weights, self.actor.model.weights, tau=1.0)

		update_target_variables(
			self.critic.target.weights, self.critic.model.weights, tau=1.0)

		self.device = '/cpu:0'
		
	def get_action(self, state):
		"""
		Predicting the action with the actor model from the state
		C2: ||Wk||<=1 so we always divide by its norm so ||Wk||==1
		"""
		is_single_state = len(state.shape) == 1

		state = np.expand_dims(state, axis=0).astype(
			np.float32) if is_single_state else state

		action = self._get_action_body(
			tf.constant(state), self.sigma)

		return action.numpy()[0] if is_single_state else action.numpy()


	@tf.function
	def _get_action_body(self, state, sigma):
		with tf.device(self.device):
			action = self.actor.model(state)
			action += tf.random.normal(shape=action.shape,
				mean=0, stddev=sigma, dtype=tf.float32)
			
			return tf.clip_by_norm(tf.clip_by_value(action, 0, self.max_action),1)


	def train_step(self, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch):
		"""
		Performing the update of the models (Actor and Critic) using the gradients on batches with size
		by default == 64
		"""
		actor_loss, critic_loss, td_error = self._train_step_body(
			states_batch, actions_batch, next_states_batch, rewards_batch, done_batch)

		if actor_loss is not None:
			tf.summary.scalar(name='Training/actor_loss', data=actor_loss)

		tf.summary.scalar(name='Training/critic_loss', data=critic_loss)

		return td_error

	@tf.function
	def _train_step_body(self, states, actions, next_states, rewards, dones):
		with tf.device(self.device):
			with tf.GradientTape() as tape:
				td_errors = self._compute_td_error_body(
					states, actions, next_states, rewards, dones)
				critic_loss = tf.reduce_mean(
					tf.square(td_errors))

			critic_grad = tape.gradient(
				critic_loss, self.critic.model.trainable_weights)
			self.critic.adam_optimizer.apply_gradients(
				zip(critic_grad, self.critic.model.trainable_weights))

			with tf.GradientTape() as tape:
				next_action = self.actor.model(states)
				actor_loss = -tf.reduce_mean(
					self.critic.model([states, next_action]))

			actor_grad = tape.gradient(
				actor_loss, self.actor.model.trainable_variables)
			self.actor.adam_optimizer.apply_gradients(
				zip(actor_grad, self.actor.model.trainable_variables))
			
			update_target_variables(
				self.actor.target.weights, self.actor.model.weights, tau=self.tow)

			update_target_variables(
				self.critic.target.weights, self.critic.model.weights, tau=self.tow)

			return actor_loss, critic_loss, td_errors

	@tf.function
	def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
		with tf.device(self.device):
			not_dones = 1. - dones
			target_Q = self.critic.target(
				[next_states, self.actor.target(next_states)])
			target_Q = rewards + (not_dones * self.gamma * target_Q)
			target_Q = tf.stop_gradient(target_Q)
			current_Q = self.critic.model([states, actions])
			td_errors = target_Q - current_Q
		return td_errors


	def train(self, action_queue, matrix_queue):

		total_steps = 0
		tf.summary.experimental.set_step(total_steps)
		episode_steps = 0
		episode_return = 0
		episode_start_time = time.perf_counter()
		n_episode = 0

		writer = tf.summary.create_file_writer('logs/'+self.logger_folder+'/'+str(self.agentId))
		writer.set_as_default()

		tf.summary.experimental.set_step(total_steps)

		replay_buffer = Replay_Buffer(
			self.max_buffer_size, self.batch_size)

		old_state = self.env.reset()
		max_reward = 0
		
		while total_steps < self.timestamp:
			if total_steps < self.n_warmup:
				action = np.random.uniform(low=0, high=1, size=self.action_dim)
				action = action/np.linalg.norm(action)
				
			else:
				action = self.get_action(old_state)


#			action = normalize(action.reshape(self.env.M, self.env.K)).reshape(self.action_dim)
			new_state, reward, done, action = self.env.step(action, self.agentId)
			if reward>max_reward:
				best_action = action
				max_reward = reward

			episode_steps += 1
			episode_return += reward
			total_steps += 1
			tf.summary.experimental.set_step(total_steps)

			replay_buffer.add_experience(old_state, action, reward, new_state, done)

			done_flag = done

			old_state = new_state

			tf.summary.scalar(name="Episode/Reward", data=reward)
			

			if done or episode_steps == self.max_epsiode_steps:
				fps = (time.perf_counter() - episode_start_time)/fps
				print("Agent: {5: 3}, Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
	                    n_episode, total_steps, episode_steps, episode_return, fps, self.agentId))
				
				n_episode+=1

				episode_steps = 0
				episode_return = 0
				episode_start_time = time.perf_counter()


				action_queue.put(best_action)
				W = matrix_queue.get()
				self.env.set_W(W)

			if total_steps < self.n_warmup:
				continue

			states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = replay_buffer.sample_batch(self.state_dim, self.action_dim)
			
			self.train_step(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch)
