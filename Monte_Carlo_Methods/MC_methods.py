import numpy as np
from collections import defaultdict

def first_visit_mc_policy_evaluation(states: list,
                                     policy: dict,
                                     generate_episode_fn,
                                     gamma = 0.99,
                                     num_episodes = 1000):
    """
    First-Visit Monte Carlo Policy Evaluation.
    
    Args:
        states (list): List of all possible states.
        policy (dict): A dictionary mapping each state to the action to take under the current policy.
        generate_episode_fn (function): A function that generates an episode following the policy.
                                        Should return a list of (state, action, reward) tuples.
        gamma (float): Discount factor.
        num_episodes (int): Number of episodes to sample.

    Returns:
        V (dict): Dictionary mapping each state to its estimated value.
    """

    # Initialize the value function for all states to 0.0
    V = {s: 0.0 for s in states}

    # initialize a list of returns for each state
    Returns = defaultdict(list)

    for episode_idx in range(num_episodes):
        # Generate an episoed: a list of (state, action, reward)
        episode = generate_episode_fn(policy)

        # Keep track of which states we have seen so far in this episode
        visited_states = set()

        G = 0
        # Process the episode in reverse to calculate returns
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = gamma * G + reward # accumulate discounted return

            if state not in visited_states:
                visited_states.add(state)
                Returns[state].append(G)
                V[state] = np.mean(Returns[state])
    
    return V

import numpy as np
from collections import defaultdict

def every_visit_mc_policy_evaluation(states,
                                     policy, 
                                     generate_episode_fn, 
                                     gamma=1.0, 
                                     num_episodes=1000):
    """
    Every-Visit Monte Carlo Policy Evaluation.

    Args:
        states (list): List of all possible states.
        actions (dict): A dictionary mapping each state to the list of available actions.
        policy (dict): A dictionary mapping each state to the action to take under the current policy.
        generate_episode_fn (function): A function that generates an episode following the policy.
                                        Should return a list of (state, action, reward) tuples.
        gamma (float): Discount factor.
        num_episodes (int): Number of episodes to sample.

    Returns:
        V (dict): Dictionary mapping each state to its estimated value.
    """
    
    # Initialize the value function
    V = {s: 0.0 for s in states}
    
    # Store all returns observed for each state
    Returns = defaultdict(list)

    for episode_idx in range(num_episodes):
        # Generate an episode: [(s0, a0, r1), (s1, a1, r2), ..., (sT-1, aT-1, rT)]
        episode = generate_episode_fn(policy)

        G = 0
        # Process the episode in reverse
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = gamma * G + reward

            # Every-Visit: no check, always update
            Returns[state].append(G)
            V[state] = np.mean(Returns[state])

    return V

def mc_control_es(states: list,
                  actions: dict,
                  generate_episode_es_fn,
                  num_episodes=10000,
                  gamma=1.0):
    """
    Monte Carlo Control with Exploring Starts (Incremental version)

    Args:
        states (list): all possible states
        actions (dict): maps state -> list of possible actions
        generate_episode_es_fn (function): returns an episode starting from a specific (s0, a0)
        num_episodes (int): number of episodes
        gamma (float): discount factor

    Returns:
        policy (dict): optimal policy found
        Q (dict): optimal action-value function
    """
    
    # Initialization
    Q = defaultdict(float)
    N = defaultdict(int) # counts of visitis per (s, a)
    policy = {s: np.random.choice(actions[s]) for s in states if actions[s]}

    for episode_idx in range(num_episodes):
        # Exploring Starts: pick random s0 and a0 such that all pairs have prob > 0
        s0_idx = np.random.randint(len(states))
        s0 = states[s0_idx]
        if not actions[s0]:
            continue
        a0 = np.random.choice(actions[s0])

        # Generate episode starting from (s0, a0)
        episode = generate_episode_es_fn(s0, a0, policy)

        G = 0
        seen_pairs = set()

        # Traverse episode backward
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in seen_pairs:
                seen_pairs.add((s, a))

                # Incremental update
                N[(s, a)] += 1
                Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

                # Policy improvement
                if actions[s]: # no terminal state
                    policy[s] = max(actions[s], key=lambda a_: Q[(s, a_)])
    
    return policy, Q