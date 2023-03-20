import random

from agent.agent import Agent
from functions import *
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.set_logical_device_configuration(
			gpus[0],
			[tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
		)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		print(e)

if len(sys.argv) != 4:
	print ("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 64

for e in range(episode_count + 1):
	print ("# Episode " + str(e) + "/" + str(episode_count) + "###############################")
	state = getState(data, 0, window_size + 1)
	end_state = random.randrange(512)
	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print ("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		done = True if t == end_state else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")
			print ("")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size, window_size)

		if done:
			break
	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
