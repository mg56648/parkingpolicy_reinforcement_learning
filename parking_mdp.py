import numpy as np
from mdp import MDP

class ParkingAction:
    m = 3
    PARK, DRIVE, EXIT = range(m)
    strings = ["PARK", "DRIVE", "EXIT"]

class ParkingMDP(MDP):
    def __init__(self, num_rows, handicap_reward=-100,
        collision_reward=-10000, not_parked_reward=-1, parked_reward_factor=10):
        """Initialization. Sets up Parking MDP.

        Args:
            num_rows: number of parking rows
        """
        self.num_rows = num_rows

        # maintain interpretable state mapping
        self.state_id_to_params = {}
        self.state_params_to_id = {}

        # set up states and actions
        # 2 parking columns * num_rows * is_occupied * is_parked + terminal_state
        self.n = 2*num_rows*2*2 + 1
        self.m = ParkingAction.m
        self.terminal_state = self.n - 1

        # set up rewards and interpretable state mappings
        self.R = np.zeros(self.n)
        self.R[self.terminal_state] = 1
        state_id_counter = 0
        for c in range(0, 2): # parking lot column
            for r in range(0, num_rows): # parking lot row
                for o in range(0, 2): # whether spot is occupied
                    for p in range(0, 2): # whether parked
                        # set up mapping between state ids and interpretable params
                        self.state_id_to_params[state_id_counter] = (c, r, o, p)
                        self.state_params_to_id[(c, r, o, p)] = state_id_counter

                        # set up rewards
                        if p == 0: # not parked
                            self.R[state_id_counter] = not_parked_reward
                        elif p == 1:
                            if o == 1: # parked in spot that is occupied
                                self.R[state_id_counter] = collision_reward
                            else:
                                if r == 0: # handicapped
                                    self.R[state_id_counter] = handicap_reward
                                else:
                                    self.R[state_id_counter] = (self.num_rows - r)*parked_reward_factor

                        state_id_counter += 1

        # initialize transitions
        self.T = []
        for a in range(self.m):
            self.T.append(np.zeros((self.n, self.n)))

        # set up transitions
        for a in range(self.m):
            for c in range(0, 2):
                for r in range(0, num_rows):
                    for o in range(0, 2):
                        for p in range(0, 2):
                            current_state = self.get_state_id(c, r, o, p)

                            if a == ParkingAction.PARK:
                                if p == 0:
                                    # park the agent
                                    next_state = self.get_state_id(c, r, o, 1)
                                    self.T[a][current_state, next_state] = 1
                                else:
                                    # terminate
                                    next_state = self.terminal_state
                                    self.T[a][current_state, next_state] = 1

                            elif a == ParkingAction.EXIT:
                                if p == 1:
                                    # exit
                                    next_state = self.terminal_state
                                    self.T[a][current_state, next_state] = 1

                            elif a == ParkingAction.DRIVE:
                                if p == 0:
                                    # determine next location of agent
                                    if c == 0:
                                        if r == 0:
                                            next_c = 1
                                            next_r = 0
                                        else:
                                            next_c = c
                                            next_r = r - 1
                                    elif c == 1:
                                        if r == self.num_rows-1:
                                            next_c = 0
                                            next_r = r
                                        else:
                                            next_c = c
                                            next_r = r + 1

                                    # next state is either occupied or not
                                    next_state1 = self.get_state_id(next_c, next_r, 0, p)
                                    next_state2 = self.get_state_id(next_c, next_r, 1, p)

                                    if r == 0: # handicap
                                        prob_occupied = 0.001
                                    else:
                                        prob_occupied = 1.*(self.num_rows-r)/self.num_rows

                                    self.T[a][current_state, next_state1] = 1-prob_occupied
                                    self.T[a][current_state, next_state2] = prob_occupied
                                else:
                                    # terminate
                                    next_state = self.terminal_state
                                    self.T[a][current_state, next_state] = 1

    def get_state_id(self, column, row, occupied, parked):
        """Gets the state id given the interpretable state params.

        Args:
            column: 0 = A, 1 = B
            row: parking row: 0, 1, 2, ..., num_rows-1
            occupied: 1 = occupied by another vehicle, 0 = otherwise
            parked: 1 = parked, 0 = otherwise

        Returns:
            state id
        """
        return self.state_params_to_id[(column, row, occupied, parked)]

    def get_state_params(self, id):
        """Gets interpretable state params given the state id.

        Args:
            id: state id

        Returns:
            tuple (column, row, occupied, parked)
            or (-1, -1, -1, -1) if terminal state
        """
        if id in self.state_id_to_params:
            (column, row, occupied, parked) = self.state_id_to_params[id]
            return (column, row, occupied, parked)
        else:
            return (-1, -1, -1, -1) # terminal state
