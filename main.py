import numpy as np

from mdp import MDP
from parking_mdp import ParkingMDP, ParkingAction
from policy import Policy, RandomParkingPolicy, SafeRandomParkingPolicy, SafeHandicapRandomParkingPolicy
from simulator import MDPSimulator
from qlearner import QLearner
from plot import Plot

from mdp_optimization import InfiniteHorizonPolicyOptimization

### SET UP
num_rows = 10
discount_factor = 0.99

# MDP 1
mdp1 = ParkingMDP(num_rows, handicap_reward=-100)
mdp1.save_to_file("MDP_parking1.txt")
initial_state1 = mdp1.get_state_id(0, num_rows-1, 0, 0)

# MDP 2
mdp2 = ParkingMDP(num_rows, handicap_reward=100)
mdp2.save_to_file("MDP_parking2.txt")
initial_state2 = mdp2.get_state_id(0, num_rows-1, 0, 0)


### PART II: MDP 1

# create simulator
simulator = MDPSimulator(mdp1, initial_state=initial_state1)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 random policy: {0}".format(avg_reward)

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 safer random policy: {0}".format(avg_reward)

# run simulation 3: safer no handicap random policy
policy = SafeHandicapRandomParkingPolicy(mdp1, park_probability=0.1, handicap_probability=0)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 safer no handicap random policy: {0}".format(avg_reward)

# run simulation 3: safer handicap random policy
policy = SafeHandicapRandomParkingPolicy(mdp1, park_probability=0.1, handicap_probability=1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 safer handicap random policy: {0}".format(avg_reward)

# run simulation 4: optimal policy
_, optimal_policy = InfiniteHorizonPolicyOptimization.policy_iteration(mdp1, discount_factor)
policy = Policy(optimal_policy)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 optimal policy: {0}".format(avg_reward)
print


### PART II: MDP 2

# create simulator
simulator = MDPSimulator(mdp2, initial_state=initial_state2)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp2, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 random policy: {0}".format(avg_reward)

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp2, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 safer random policy: {0}".format(avg_reward)

# run simulation 3: safer no handicap random policy
policy = SafeHandicapRandomParkingPolicy(mdp2, park_probability=0.1, handicap_probability=0)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 safer no handicap random policy: {0}".format(avg_reward)

# run simulation 3: safer handicap random policy
policy = SafeHandicapRandomParkingPolicy(mdp2, park_probability=0.1, handicap_probability=1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 safer handicap random policy: {0}".format(avg_reward)

# run simulation 4: optimal policy
_, optimal_policy = InfiniteHorizonPolicyOptimization.policy_iteration(mdp2, discount_factor)
policy = Policy(optimal_policy)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 optimal policy: {0}".format(avg_reward)
print


### SETUP
num_learning_trials = 10000
num_simulation_trials = 1000
num_learning_epochs = 15


### PART III: MDP 1 epsilon experiments
epsilon_list = [0.1, 0.25, 0.5, 0.75]
learning_rate = 0.01
epoch_list = []
avg_reward_list = []

for e, epsilon in enumerate(epsilon_list):
    print "Epsilon: {0}".format(epsilon)

    qlearner = QLearner(mdp1, initial_state1, epsilon=epsilon, alpha=learning_rate)

    epoch_list.append(range(num_learning_epochs))
    avg_reward_list.append([])
    for epoch in epoch_list[e]:
        for trial in range(num_learning_trials):
            qlearner.run_learning_trial()

        avg_reward = 0
        for trial in range(num_simulation_trials):
            (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
            avg_reward += total_reward
        avg_reward = 1.*avg_reward/num_simulation_trials
        avg_reward_list[e].append(avg_reward)
        print "MDP1 epoch {0}: {1}".format(epoch, avg_reward)

Plot.plot_multiple(epoch_list, avg_reward_list, [str(e) for e in epsilon_list], 'epsilon', 'MDP1 Learning: Epsilon', 'mdp1_epsilon_plot.png')
print


### PART III: MDP 1 alpha experiments
epsilon = 0.25
learning_rate_list = [0.001, 0.01, 0.1, 1.0]
epoch_list = []
avg_reward_list = []

for a, learning_rate in enumerate(learning_rate_list):
    print "Alpha: {0}".format(learning_rate)

    qlearner = QLearner(mdp1, initial_state1, epsilon=epsilon, alpha=learning_rate)

    epoch_list.append(range(num_learning_epochs))
    avg_reward_list.append([])
    for epoch in epoch_list[a]:
        for trial in range(num_learning_trials):
            qlearner.run_learning_trial()

        avg_reward = 0
        for trial in range(num_simulation_trials):
            (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
            avg_reward += total_reward
        avg_reward = 1.*avg_reward/num_simulation_trials
        avg_reward_list[a].append(avg_reward)
        print "MDP1 epoch {0}: {1}".format(epoch, avg_reward)

Plot.plot_multiple(epoch_list, avg_reward_list, [str(a) for a in learning_rate_list], 'alpha', 'MDP1 Learning: Learning Rate', 'mdp1_alpha_plot.png')
print


### PART III: MDP 2 epsilon experiments
epsilon_list = [0.1, 0.25, 0.5, 0.75]
learning_rate = 0.01
epoch_list = []
avg_reward_list = []

for e, epsilon in enumerate(epsilon_list):
    print "Epsilon: {0}".format(epsilon)

    qlearner = QLearner(mdp2, initial_state2, epsilon=epsilon, alpha=learning_rate)

    epoch_list.append(range(num_learning_epochs))
    avg_reward_list.append([])
    for epoch in epoch_list[e]:
        for trial in range(num_learning_trials):
            qlearner.run_learning_trial()

        avg_reward = 0
        for trial in range(num_simulation_trials):
            (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
            avg_reward += total_reward
        avg_reward = 1.*avg_reward/num_simulation_trials
        avg_reward_list[e].append(avg_reward)
        print "MDP2 epoch {0}: {1}".format(epoch, avg_reward)

Plot.plot_multiple(epoch_list, avg_reward_list, [str(e) for e in epsilon_list], 'epsilon', 'MDP2 Learning: epsilon', 'mdp2_epsilon_plot.png')
print


### PART III: MDP 2 alpha experiments
epsilon = 0.25
learning_rate_list = [0.001, 0.01, 0.1, 1.0]
epoch_list = []
avg_reward_list = []

for a, learning_rate in enumerate(learning_rate_list):
    print "Alpha: {0}".format(learning_rate)

    qlearner = QLearner(mdp2, initial_state2, epsilon=epsilon, alpha=learning_rate)

    epoch_list.append(range(num_learning_epochs))
    avg_reward_list.append([])
    for epoch in epoch_list[a]:
        for trial in range(num_learning_trials):
            qlearner.run_learning_trial()

        avg_reward = 0
        for trial in range(num_simulation_trials):
            (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
            avg_reward += total_reward
        avg_reward = 1.*avg_reward/num_simulation_trials
        avg_reward_list[a].append(avg_reward)
        print "MDP2 epoch {0}: {1}".format(epoch, avg_reward)

Plot.plot_multiple(epoch_list, avg_reward_list, [str(a) for a in learning_rate_list], 'alpha', 'MDP2 Learning: Learning Rate', 'mdp2_alpha_plot.png')
print
