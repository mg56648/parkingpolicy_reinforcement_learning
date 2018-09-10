import numpy as np
from mdp import MDP

class InfiniteHorizonPolicyEvaluation:
    """Infinite horizon policy evaluation for MDPs.
    """

    @staticmethod
    def direct_solver(mdp, policy, beta):
        """Policy evaluation using linear algebra.

        Args:
            mdp: Markov Decision Process object
            policy: policy to evaluate (n-dimensional vector)
            beta: discount factor [0, 1)
        """
        I = np.eye(mdp.n)
        T = np.zeros((mdp.n, mdp.n))
        for s in range(mdp.n):
            a = policy[s]
            T[s,:] = mdp.get_transition_prob(s, a)
        R = mdp.get_reward()
        return np.dot(np.linalg.inv( I - beta*T ), R)

class InfiniteHorizonPolicyOptimization:
    """Infinite horizon policy optimization for MDPs.
    """

    @staticmethod
    def policy_iteration(mdp, beta):
        """Policy iteration optimization.

        Args:
            mdp: Markov Decision Process object
            beta: discount factor [0, 1)

        Returns:
            (optimal non-stationary value function, non-stationary policy)
            for the MDP and discount factor
        """
        # initialize policy to random policy
        policy = np.random.randint(0, mdp.m, (mdp.n,))

        # loop
        while True:
            # evaluate policy
            V = InfiniteHorizonPolicyEvaluation.direct_solver(mdp, policy, beta)

            # improve policy
            improved_policy = InfiniteHorizonPolicyOptimization.improve_policy(V, mdp)

            # replace policy with new one until no longer can improve
            if np.all(policy == improved_policy):
                break
            policy = np.array(improved_policy)

        return (V, policy)

    @staticmethod
    def improve_policy(V, mdp):
        """Helper for policy iteration.

        Args:
            V: value function
            mdp: Markov Decision Process object

        Returns:
            improved policy based on value function
        """
        new_policy = np.zeros(mdp.n, dtype=int)
        for s in range(mdp.n):
            expectations = [np.dot(mdp.get_transition_prob(s, a), V) for a in range(mdp.m)]
            new_policy[s] = np.argmax(expectations)
        return new_policy

    @staticmethod
    def value_iteration(mdp, beta, epsilon=0.000001):
        """Infinite-horizon value iteration optimization.

        Args:
            mdp: Markov Decision Process object
            beta: discount factor [0, 1)
            epsilon: difference parameter for stopping condition

        Returns:
            (optimal non-stationary value function, non-stationary policy)
            for the MDP and discount factor
        """
        V = np.zeros((mdp.n,))
        V_prev = np.array(V)
        policy = np.zeros((mdp.n,), dtype=int)

        # iterative case
        MAX_ITERATIONS = 100000
        for i in range(MAX_ITERATIONS):
            for s in range(mdp.n):
                expectations = [np.dot(mdp.get_transition_prob(s, a), V_prev) for a in range(mdp.m)]
                V[s] = mdp.get_reward(s) + beta * np.max(expectations)

            stopping_condition = np.linalg.norm(V_prev - V, ord=np.inf)
            if stopping_condition <= epsilon and i > 0:
                print "(converged in {} iterations)".format(i)
                break
            V_prev = np.array(V)
        if i >= MAX_ITERATIONS:
            print "(may or may not have converged in {} max iterations)".format(i)

        # get policy
        for s in range(mdp.n):
            expectations = [np.dot(mdp.get_transition_prob(s, a), V) for a in range(mdp.m)]
            policy[s] = np.argmax(expectations)

        return (V, policy)
