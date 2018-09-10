import numpy as np

from parking_mdp import ParkingAction

class Policy:
    """Policy from policy vector.
    """
    def __init__(self, policy):
        """Initialization.

        Args:
            policy: policy vector
        """
        self.policy = policy

    def get_action(self, state):
        """Get the policy action for the given state.

        Args:
            state: state

        Returns:
            action
        """
        return self.policy[state]

class RandomParkingPolicy(Policy):
    """Random policy.

    Selects PARK with probability p and DRIVE with probability 1-p.
    """
    def __init__(self, mdp, park_probability=0.5):
        """Initialization.

        Args:
            mdp: MDP object
            park_probability: probability of parking [0,1]
        """
        self.mdp = mdp
        self.park_probability = park_probability

    def get_action(self, state):
        (column, row, occupied, parked) = self.mdp.get_state_params(state)

        if parked == 1:
            action = ParkingAction.EXIT
        elif np.random.uniform() < self.park_probability:
            action = ParkingAction.PARK
        else:
            action = ParkingAction.DRIVE

        return action

class SafeRandomParkingPolicy(Policy):
    """Safer random policy.

    If occupied, selects DRIVE. Otherwise:
    Selects PARK with probability p and DRIVE with probability 1-p.
    """
    def __init__(self, mdp, park_probability=0.5):
        """Initialization.

        Args:
            mdp: MDP object
            park_probability: probability of parking [0,1]
        """
        self.mdp = mdp
        self.park_probability = park_probability

    def get_action(self, state):
        (column, row, occupied, parked) = self.mdp.get_state_params(state)

        if parked == 1:
            action = ParkingAction.EXIT
        elif occupied == 1:
            action = ParkingAction.DRIVE
        elif occupied == 0 and np.random.uniform() < self.park_probability:
            action = ParkingAction.PARK
        else:
            action = ParkingAction.DRIVE

        return action

class SafeHandicapRandomParkingPolicy(Policy):
    """Safer random policy with probability of parking in handicap spot.

    If occupied, selects DRIVE. Otherwise:
    If handicap, selects PARK with probability p_h and DRIVE with probability 1-p_h.
    Otherwise selects PARK with probability p and DRIVE with probability 1-p.
    """
    def __init__(self, mdp, park_probability=0.5, handicap_probability=0):
        """Initialization.

        Args:
            mdp: MDP object
            park_probability: probability of parking [0,1]
            handicap_probability: probability of parking when at handicap [0,1]
        """
        self.mdp = mdp
        self.park_probability = park_probability
        self.handicap_probability = handicap_probability

    def get_action(self, state):
        (column, row, occupied, parked) = self.mdp.get_state_params(state)

        if parked == 1:
            action = ParkingAction.EXIT
        elif occupied == 1:
            action = ParkingAction.DRIVE
        elif occupied == 0:
            if row == 0:
                if np.random.uniform() < self.handicap_probability:
                    action = ParkingAction.PARK
                else:
                    action = ParkingAction.DRIVE
            elif np.random.uniform() < self.park_probability:
                action = ParkingAction.PARK
            else:
                action = ParkingAction.DRIVE

        return action
