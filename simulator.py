import numpy as np

from mdp import MDP

class MDPSimulator:
    def __init__(self, mdp, initial_state=0):
        """Initialization.

        Args:
            mdp: MDP object
            intial_state: initial state
        """
        self.mdp = mdp
        self.reset(initial_state=initial_state)

    def reset(self, initial_state=0):
        """Reset simulator.

        Args:
            initial_state: initial state
        """
        self.current_state = initial_state
        self.terminal_state = False

    def run_simulation(self, policy):
        """Run simulation with policy until terminal state.

        Args:
            policy: policy object

        Returns:
            (total_reward, state_seq, action_seq) where
                total_reward is total accumulated reward
                state_seq is sequence of states
                action_seq is sequence of actions taken
        """
        total_reward = 0
        state_seq = []
        action_seq = []
        while not self.in_terminal_state():
            # get state and reward
            state = self.get_current_state()
            state_seq.append(state)
            reward = self.get_reward()
            total_reward += reward

            # use provided policy
            action = policy.get_action(state)
            action_seq.append(action)

            # take action
            self.take_action(action)

        return (total_reward, state_seq, action_seq)

    def take_action(self, action):
        """Take an action to go to a new state.

        Args:
            action: action
        """
        if self.terminal_state:
            return

        # set up distribution to sample
        values = np.array(range(self.mdp.n))
        probabilities = self.mdp.get_transition_prob(self.current_state, action)
        if np.sum(probabilities) == 0:
            return
        bins = np.add.accumulate(probabilities)

        # sample the next state from the distribution and update
        next_state = values[np.digitize(np.random.random_sample(1), bins)[0]]
        self.current_state = next_state

        # check if in terminal state
        self.terminal_state = True
        for a in range(self.mdp.m):
            probabilities = self.mdp.get_transition_prob(self.current_state, a)
            if np.sum(probabilities) > 0:
                self.terminal_state = False
                break

    def in_terminal_state(self):
        """Check if in a terminal state.
        A terminal state has no transitions out in any action.

        Returns:
            if in terminal state
        """
        return self.terminal_state

    def get_current_state(self):
        """Get current state.

        Returns:
            current state
        """
        return self.current_state

    def get_reward(self):
        """Get reward of current state.

        Returns:
            reward of current state
        """
        return self.mdp.get_reward(self.current_state)
