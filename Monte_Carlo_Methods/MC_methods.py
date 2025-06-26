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
                  policy = None,
                  num_episodes=10000,
                  gamma=1.0):
    """
    Monte Carlo Control with Exploring Starts (Incremental version)

    Args:
        states (list): all possible states
        actions (dict): maps state -> list of possible actions
        generate_episode_es_fn (function): returns an episode starting from a specific (s0, a0)
        policy (dict): initial policy
        num_episodes (int): number of episodes
        gamma (float): discount factor

    Returns:
        policy (dict): optimal policy found
        Q (dict): optimal action-value function
    """
    # Si no hay policy, creamos una de forma arbitraria.
    if policy == None:
        policy = {}
        for s in states:
            # Cada state tiene una posibilidad de acciones --> elegimos una aleatoriamente
            policy[s] = np.random.choice(actions[s])
    # Creamos una value functions arbitraria, con todo 0's. Se tratará de un diccionario de diccionarios: {state_i : {action_1:value_1, action_2:value_2}}
    Q = {}
    for s in states:
        Q[s] = {} # Creamos el diccionario interior para el state s
        for a in actions[s]:
            Q[s][a] = 0.0 # Inicializamos con el valor Q(s, a) con 0.0
    # Inicializamos un nuevo diccionario de diccionarios Returns(s, a)
    returns = defaultdict(lambda: defaultdict(list))
    # Iteramos sobre cada episodio
    for _ in range(num_episodes):
        # Inicializamos un conjunto para comprobar si ya hemos visitado el par (state, action)
        visited_state_action_pairs = set()
        # Elegimos de forma aleatoria el state y la action con la que vamos a iniciar el episodio
        s0 = np.random.choice(states)
        a0 = np.random.choice(actions[s])
        # Generamos el episodio --> [(s0, a0, r1), (s1, a1, r2), ..., (sT-1, aT-1, rT)]
        episode = generate_episode_es_fn(s0, a0, policy)
        # Inicializamos G
        G = 0
        for t in reversed(range(len(episode))):
            # Actualizamos G
            state, action, reward = episode[t]
            G = gamma * G + reward
            # Si el par state-action se visita 
            if (state, action) in visited_state_action_pairs:
                continue
            # Añadimos el par state-action cuando se visita por primera vez
            visited_state_action_pairs.add((state, action))
            # Añadimos el valor del return en ese momento
            returns[state][action].append(G)
            # Actualizamos el valor de Q (value function)
            Q[state][action] = np.mean(returns[state][action])
            # Hacemos la policy greedy respecto a este estado
            best_action = max(Q[state], key=Q[s].get)
            policy[state] = best_action
    return Q, policy