import random


def select_random(probabilities):
    """
    Selects random index according to the
    assigned probabilities
    """
    assert abs(sum(probabilities) - 1) < 1e-9
    n = random.random()
    total = 0
    index = 0
    for prob in probabilities:
        total += prob
        if n < total:
            return index
        index = index + 1


class MDP:
    """
    Creates a MDP, given no of states, actions, transition matrix, reward matrix
    gamma is something left to the user to implement

    States -> no of states
    Actions -> no of actions
    transition matrix -> T[s][a][s'](expect this to be a numpy array)
    Reward matrix -> R[s][a](expect this to be a numpy array)
    """

    def __init__(self, states, actions, transition_matrix, reward_matrix, init_state=0):
        self.S = states
        self.A = actions
        assert transition_matrix.shape == (states, actions, states)
        assert reward_matrix.shape == (states, actions)
        self.T = transition_matrix
        self.R = reward_matrix
        self.current_state = init_state

    def is_valid(self, action):
        transition_prob = self.T[self.current_state][action]
        if abs(sum(transition_prob) - 1) < 1e-9:
            return True
        else:
            return False

    def get_model(self):
        return {"Rewards": self.R,
                "Transitions": self.T}

    def play_move(self, action):
        if self.is_valid(action):
            transition_prob = self.T[self.current_state][action]
            next_state = select_random(transition_prob)
            reward = self.R[self.current_state][action]
            self.current_state = next_state
            return next_state, reward
        else:
            print('Please enter a valid action')

    def get_description(self):
        return {"States": self.S,
                "Actions": self.A}

    def get_current_state(self):
        return self.current_state

    def reset_state(self):
        self.current_state = 0

    def get_valid_actions(self):
        valid_actions = []
        for i in range(self.A):
            if self.is_valid(i):
                valid_actions.append(i)
        return valid_actions

    def set_state(self, state):
        self.current_state = state

    """
    get samples returns u a few samples of next states and rewards from
    the current state given u take the action a
    """

    def get_samples(self, state, action, n):
        prev_state = self.current_state
        self.current_state = state
        states = []
        rewards = []
        for i in range(n):
            (next_state, reward) = self.play_move(action)
            states.append(next_state)
            rewards.append(reward)
            self.current_state = state

        self.current_state = prev_state
        return states, rewards
