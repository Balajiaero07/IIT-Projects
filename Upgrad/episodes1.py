
# coding: utf-8

# In[199]:


import numpy as np
import random
import math


m = 10 # number of locations, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 30 # number of days, ranges from 0 ... d-1
episode_length = 10
Time_matrix = np.random.randint(1, 11,(m, m))

def reward_func(state, action, time_matrix):
	start_loc, time, day = state
	pickup, drop = action
	if pickup == 0 and drop == 0:
		return -1
	else:
		return time_matrix[pickup - 1, drop - 1] - time_matrix[start_loc - 1, pickup - 1]

def next_state_func(state, action, time_matrix, t, d):
	start_loc, time, day = state
	pickup, drop = action
	if pickup == 0 and drop == 0:
		time_elapsed = 1
		drop = start_loc
	else:
		time_elapsed = time_matrix[start_loc - 1, pickup - 1] + time_matrix[pickup - 1, drop - 1]

	time_next = (time + time_elapsed) % t
	day_next = (day + (time + time_elapsed)//t) % d

	return drop, time_next, day_next

# code for genrating episode
current_state =  (np.random.randint(1, m+1),np.random.randint(0, t),np.random.randint(0, d))
for epi_len in range(episode_length):
    # pick a random action
    action = (np.random.randint(0, m), np.random.randint(0, m))
    if action[0] == 0 or action[1] == 0:
    	action = (0,0)

    reward = reward_func(current_state, action, Time_matrix)
    next_state = next_state_func(current_state, action, Time_matrix, t, d)
    print("State: ",current_state, " Action: ", action, " Reward: ", reward, " Nextstate: ", next_state)
    current_state = next_state

# In[198]:




