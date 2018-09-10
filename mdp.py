import numpy as np
from io import StringIO

class IOState:
    Ready, Reading = range(2)

class MDP:
    """Markov Decision Process: {S, A, R, T}
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to empty MDP.
        """
        # number of states
        self.n = 0
        # number of actions
        self.m = 0
        # transitition function
        # list (size m) of nxn matrices
        self.T = []
        # reward function
        # vector of rewards
        self.R = np.array([])

    def load_from_file(self, filename):
        """Load MDP from file.

        Args:
            filename: path to MDP file
        """
        self.reset()
        transition_read_status = IOState.Ready
        current_transition_string = u""
        transition_counter = 0
        with open(filename, 'r') as f:
            reading_transition_matrix = 0
            for linenum, line in enumerate(f):
                if linenum == 0:
                    # read in n, m
                    n, m = line.decode("utf-8-sig").encode("utf-8").split(' ')
                    self.n = int(n)
                    self.m = int(m)
                    last_line_num = 2+self.m*(self.n+1)
                elif linenum > 1 and linenum < last_line_num:
                    if transition_read_status == IOState.Ready:
                        # read in first line of matrix
                        current_transition_string += line
                        transition_counter += 1
                        transition_read_status = IOState.Reading
                    elif transition_read_status == IOState.Reading and transition_counter < self.n:
                        # keep constructing matrix string
                        current_transition_string += line
                        transition_counter += 1
                    elif transition_read_status == IOState.Reading and transition_counter == self.n:
                        # convert finished string into numpy array
                        c = StringIO(current_transition_string)
                        T = np.loadtxt(c)
                        self.T.append(T)

                        # reset state/counters
                        transition_read_status = IOState.Ready
                        current_transition_string = u""
                        transition_counter = 0
                elif linenum == last_line_num:
                    # read in the rewards
                    c = StringIO(u""+line)
                    self.R = np.loadtxt(c)

    def save_to_file(self, filename):
        """Save MDP to file.

        Args:
            filename: path to MDP file
        """
        with open(filename, 'w') as f:
            f.write('{} {}\n\n'.format(self.n, self.m))
            for a in range(self.m):
                matrix = '\n'.join('    '.join('{0:0.8f}'.format(float(c)) for c in r) for r in self.T[a])
                f.write('{}\n\n'.format(matrix))
            f.write('    '.join(['{0:0.8f}'.format(float(a)) for a in self.R]))
            f.write('\n')

    def get_transition_prob(self, state, action, next_state=None):
        """Get transition probabilities given current state and action.

        Args:
            state: integer index of current state
            action: integer index of action
            next_state: integer index of next action

        Returns:
            transition probability given the current state, action and next
                state
            if the next_state is None then returns the transition probabilities
                given the current state and action
        """
        if next_state is None:
            return self.T[action][state, :]
        else:
            return self.T[action][state, next_state]

    def get_reward(self, state=None):
        """Get reward for state.

        Args:
            state: integer index of state

        Returns:
            reward for state
            if state is None then returns the rewards of all states as vector
        """
        if state is None:
            return self.R
        else:
            return self.R[state]

    def get_num_states(self):
        """Get number of states.

        Returns:
            number of states
        """
        return self.n

    def get_num_actions(self):
        """Get number of actions.

        Returns:
            number of actions
        """
        return self.m

    # properties
    n = property(get_num_states)
    m = property(get_num_actions)
