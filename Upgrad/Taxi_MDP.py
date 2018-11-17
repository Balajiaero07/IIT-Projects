"""
This file is specific to taxi problem.
Cant think of a way to create a generic large state MDP
which could be used in taxi problem
"""
import numpy as np
from MDP import MDP
"""
Creating the basic taxi MDP

State XiTjXk is represented by the state 24*7*i + 7*j + k
Actions are move(it has 7 components), wait, accept, reject
Transition Matrix = 1176*10*1176 (some actions not valid in some states)
Reward Matrix = 1176*10 (modeled as a function of s,a)
"""

L_i = 7
T_i = 24
U = 10
g = 35
c = 5
w = 0
States = L_i * T_i * L_i
Actions = L_i + 3  # move for all locations, wait, accept, reject
Time_matrix = np.random.randint(1, T_i, (L_i, L_i))
T = np.zeros((States, Actions, States))
R = np.zeros((States, Actions))  # modeled independent of final state
alpha = np.zeros((L_i, T_i, L_i))

for i in range(L_i):
	for j in range(T_i):
		for k in range(L_i):
			if i == k:								# in states of type A2A, B3B etc
				for m in range(L_i):
					if m != i:
						T[i*T_i*L_i + j*L_i + k][m][m*T_i*L_i + ((j+Time_matrix[i][m]) % T_i)*L_i + m] = 1     # for all move actions
						R[i*T_i*L_i + j*L_i + k][m] = -c*Time_matrix[i][m]
				# for waiting and getting to next epoch
				R[i*T_i*L_i + j*L_i + k][L_i] = -w
				T[i*T_i*L_i + j*L_i + k][L_i][i*T_i*L_i + ((j+1) % T_i)*L_i + k] = np.random.rand()
				# for waiting and getting a ride
				for m in range(1, L_i):
						T[i*T_i*L_i + j*L_i + k][L_i][i*T_i*L_i + j*L_i + ((k+m) % L_i)] = np.random.rand()
				T[i*T_i*L_i + j*L_i + k][L_i] /= np.sum(T[i*T_i*L_i + j*L_i + k][L_i])
				for m in range(L_i):
					if i == m:
						alpha[i, j, m] = T[i*T_i*L_i + j*L_i + k][L_i][i*T_i*L_i + ((j+1) % T_i)*L_i + i]
					else:
						alpha[i, j, m] = T[i*T_i*L_i + j*L_i + k][L_i][i*T_i*L_i + j*L_i + m]
			else:
				# for accepting ride
				T[i*T_i*L_i + j*L_i + k][L_i + 1][k*T_i*L_i + ((j+Time_matrix[i][k]) % T_i)*L_i + k] = 1
				R[i*T_i*L_i + j*L_i + k][L_i + 1] = g*Time_matrix[i][k]
				# for rejecting ride
				T[i*T_i*L_i + j*L_i + k][L_i + 2][i*T_i*L_i + j*L_i + i] = 1
				R[i*T_i*L_i + j*L_i + k][L_i + 2] = -U

Taxi_MDP = MDP(States, Actions, T, R)
# print(T)

# for i in range(States):
# print(np.sum(T[i], axis=1))
"""
def print_description(state, L_i, T_i):
	start_state = state/(L_i*T_i)
	time = (state/L_i)%T_i + 1
	end_state = state%L_i
	print(chr(65+start_state) + str(time) + chr(65+end_state))

def print_action(action, L_i):
	if action < L_i:
		print("move ",chr(65+action%L_i))
	elif action == L_i:
		print("wait")
	elif action == L_i+1:
		print("accept")
	else:
		print("reject")

for i in range(100):
	curr_state = Taxi_MDP.get_current_state()
	print_description(curr_state, L_i, T_i)
	while True:
		act = np.random.randint(0, Actions)
		if Taxi_MDP.is_valid(act):			
			print_action(act, L_i)
			(next_state, reward) = Taxi_MDP.play_move(act)
			print(reward)
			break
"""