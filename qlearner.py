import numpy as np
import random
import time

from simulator import MDPSimulator

class QLearner:
    """Q-learning algorithm.
    """
    def __init__(self, mdp, initial_state, epsilon=0.1, alpha=0.01):
        """Initialization.

        Args:
            mdp: MDP object
            initial_state: initial state
        """
        self.simulator = MDPSimulator(mdp, initial_state=initial_state)
        self.initial_state = initial_state

        # Q-table
        self.qtable = np.zeros((mdp.n, mdp.m))
        self.num_learning_trials = 0

        # parameters for Q-learning
        self.alpha = alpha # learning rate
        self.beta = 0.99 # discount factor

        # parameters for explore-exploit policy
        self.epsilon = epsilon # epsilon-greedy: probability select random action

        # other parameters
        self.MAX_TRIAL_NUM_STEPS = 100

    def run_simulation_trial(self):
        """Run one trial of simulation using Q-table without any learning.
        """
        total_reward = 0
        state_seq = []
        action_seq = []
        step = 0
        self.simulator.reset(initial_state=self.initial_state)
        while not self.simulator.in_terminal_state() and step < self.MAX_TRIAL_NUM_STEPS:
            # get state and reward
            state = self.simulator.get_current_state()
            state_seq.append(state)
            reward = self.simulator.get_reward()
            total_reward += reward

            # get action from Q-table
            action = np.argmax(self.qtable[state,:])
            action_seq.append(action)

            # take action
            self.simulator.take_action(action)

            step += 1

        return (total_reward, state_seq, action_seq)

    def run_learning_trial(self):
        """Run one trial of learning.
        """
        state_seq = []
        reward_seq = []
        action_seq = []
        step = 0
        self.simulator.reset(initial_state=self.initial_state)
        while not self.simulator.in_terminal_state() and step < self.MAX_TRIAL_NUM_STEPS:
            # get state and reward
            state = self.simulator.get_current_state()
            state_seq.append(state)
            reward = self.simulator.get_reward()
            reward_seq.append(reward)

            # get action from explore-exploit policy
            action = self.explore_exploit_policy(state)
            action_seq.append(action)

            # take action
            self.simulator.take_action(action)

            step += 1

        # add final state to state sequence
        state = self.simulator.get_current_state()
        state_seq.append(state)
        reward = self.simulator.get_reward()
        reward_seq.append(reward)

        # update
        self.do_reverse_q_updates(state_seq, reward_seq, action_seq)
        self.num_learning_trials += 1

    def explore_exploit_policy(self, current_state):
        """Epsilon-greedy explore-exploit policy.

        Args:
            current_state: current state

        Returns:
            action to take
        """
        if self.num_learning_trials == 0 or random.uniform(0, 1) <= self.epsilon:
            num_actions = self.qtable.shape[1]
            return random.randint(0, num_actions-1)
        else:
            return np.argmax(self.qtable[current_state,:])

    def do_reverse_q_updates(self, state_seq, reward_seq, action_seq):
        """Perform reverse updates given trajectory.

        Args:
            state_seq: sequence of states
            reward_seq: sequence of rewards, 1:1 with state_seq
            action_seq: sequence of actions, action_seq[i] yields state_seq[i+1]
        """
        assert(len(state_seq) == len(action_seq) + 1)
        assert(len(state_seq) == len(reward_seq))

        for i in reversed(range(len(action_seq))):
            state = state_seq[i]
            action = action_seq[i]
            new_state = state_seq[i+1]
            reward = reward_seq[i]
            self.qtable[state, action] = self.compute_q_update(state, action, new_state, reward)

    def compute_q_update(self, old_state, old_action, new_state, reward):
        """Computes the Q-update.

        Args:
            old_state: old state
            old_action: old action taken from old state to get to new state
            new_state: new state
            reward: reward from old state

        Returns:
            Q-update
        """
        num_actions = self.qtable.shape[1]
        old_value = self.qtable[old_state, old_action]
        optimal_future_estimate = np.max([self.qtable[new_state, a] for a in range(num_actions)])
        learned_value = reward + self.beta * optimal_future_estimate
        update = self.alpha * (learned_value - old_value)
        return old_value + update
