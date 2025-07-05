import numpy as np
from collections import defaultdict

def first_visit_mc_policy_evaluation(states: list,
                                     policy: dict,
                                     generate_episode_fn,
                                     gamma = 0.99,
                                     num_episodes = 1000):
    """
    First-Visit Monte Carlo Policy Evaluation - Incremental Implementation.
    
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

    # Initialize a count for each state s
    N = {}
    for s in states:
        N[s] = 0

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
                N[state] = N[state] + 1
                V[state] = V[state] + (G - V[state]) / N[state]
    
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
        generate_episode_fn (function): A function that generates an episode following the policy (deterministic).
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
        generate_episode_es_fn (function): returns an episode starting from a specific (s0, a0) and given a deterministic policy
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
        states_idx = np.random.choice(range(len(states)))
        s0 = states[states_idx]
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
    # Construimos la policy óptima
    optimal_policy = {}
    for s in states:
        best_action = max(Q[state], key=Q[s].get)
        optimal_policy[s] = best_action
    return Q, optimal_policy

def mc_control_on_policy_first_visit(states: list,
                                     actions: dict,
                                     generate_episode_fn_with_soft_policy,
                                     epsilon = 0.01,
                                     num_episodes = 100000,
                                     gamma = 1):
    '''
    Monte Carlo control without exploring starts, insetad we use an epsilon-greedy policy

    Args:
        states (list): all possible states
        actions (dict): maps state -> list of possible actions
        generate_episode_es_fn (function): returns an episode given a policy
        epsilon (float): probability of not choosing directly the greedy action
        num_episodes (int): number of episodes
        gamma (float): discount factor

    Returns:
        policy (dict): optimal policy found
        Q (dict): optimal action-value function
    '''
    # Define an arbitrary epsilon-soft policy
    soft_policy = {}
    for s in states:
        soft_policy[s] = {}
        n_actions = len(actions[s])
        # Elegimos una acción greedy arbitraria
        best_action = np.random.choice(actions[s])
        for a in actions[s]:
            if a == best_action:
                soft_policy[s][a] = 1 - epsilon + (epsilon / n_actions)
            else:
                soft_policy[s][a] = epsilon / n_actions
    # Creamos una value functions arbitraria, con todo 0's. Se tratará de un diccionario de diccionarios: {state_i : {action_1:value_1, action_2:value_2}}
    Q = {}
    for s in states:
        Q[s] = {} # Creamos el diccionario interior para el state s
        for a in actions[s]:
            Q[s][a] = 0.0 # Inicializamos con el valor Q(s, a) con 0.0
    # Inicializamos un nuevo diccionario de diccionarios Returns(s, a)
    returns = defaultdict(lambda: defaultdict(list))
    for _ in range(num_episodes):
        # Inicializamos un conjunto para comprobar si ya hemos visitado el par (state, action)
        visited_state_action_pairs = set()
        # Generamos el episodio --> [(s0, a0, r1), (s1, a1, r2), ..., (sT-1, aT-1, rT)]
        episode = generate_episode_fn_with_soft_policy(soft_policy)
        G = 0
        for t in reversed(range(len(episode))):
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
            # Elegimos la acción greedy, y si hay empate elegimos de forma random
            q_vals = Q[state]
            max_value = max(q_vals.values())
            best_actions = [a for a in q_vals if q_vals[a] == max_value]
            best_action = np.random.choice(best_actions)
            num_actions = len(actions[state])
            for a in actions[state]:
                if a == best_action:
                    soft_policy[state][a] = 1 - epsilon + (epsilon / num_actions)
                else:
                    soft_policy[state][a] = epsilon / num_actions

def mc_policy_evaluation_off_policy_every_visit(states:list,
                                                actions: dict,
                                                target_policy: dict,
                                                generate_episode_fn_behavior_policy,
                                                gamma = 1.0,
                                                num_episodes = 10000):
    '''
    Monte Carlo for policy evaluation build with an off-policy weighted importance-sampling incremental implementation every-visit method.

    Args:
        states (list): List of all possible states.
        actions (dict): A dictionary mapping each state to the list of available actions.
        target_policy (dict{dict}): A dictionary of diccionaries mapping each state to the probability of each possible action to take under the current policy.
        generate_episode_fn_behavior_policy (function): takes a policy 'b' (behavior policy) and returns a list of tuples [(s0, a0, r1), (s1, a1, r2), ..., (sT-1, aT-1, rT)].
        gamma (float): Discount factor.
        num_episodes (int): Number of episodes to sample.

    Returns:
        Q (dict{dict}): Dictionary of diccionaries mapping each state-action pair to its estimated value.
    '''
    # Initialize Q (value function) and C (sum of weights) 
    Q = defaultdict(lambda: defaultdict(int))
    C = {}
    for s in states:
        C[s] = {}
        for a in actions[s]:
            C[s][a] = 0
    # Loop for each episode
    for _ in range(num_episodes):
        # Create behavior policy - complete random - dictionary of dictionaries
        behavior_policy = {}
        for s in states:
            n = len(actions[s])
            probs = np.random.dirichlet(np.ones(n))
            behavior_policy[s] = {a: p for a, p in zip(actions[s], probs)}
        # Generate an episode
        episode = generate_episode_fn_behavior_policy(b)
        # Initialize G and W
        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            if W != 0:
                G = gamma * G + reward
                C[state][action] = C[state][action] + W
                Q[state][action] = Q[state][action] + (W / C[state][action]) * (G - Q[state][action])
                W = W * (target_policy[state][action] / behavior_policy[state][action])
            else:
                break
    return Q

def mc_control_off_policy_every_visit(states:list,
                                    actions: dict,
                                    generate_episode_fn_behavior_policy,
                                    gamma = 1.0,
                                    num_episodes = 10000):
    '''
    Monte Carlo for Control (estimating optimal policy) build with an off-policy weighted importance-sampling incremental implementation every-visit method.

    Args:
        states (list): List of all possible states.
        actions (dict): A dictionary mapping each state to the list of available actions.
        generate_episode_fn_behavior_policy (function): takes a policy 'b' (behavior policy) and returns a list of tuples [(s0, a0, r1), (s1, a1, r2), ..., (sT-1, aT-1, rT)].
        gamma (float): Discount factor.
        num_episodes (int): Number of episodes to sample.

    Returns:
        Q (dict{dict}): Dictionary of diccionaries mapping each state-action pair to its estimated value.
        target_policy (dict): A deterministic policy, mapping each state to an action.
    '''
    # Initialize the value-function Q, C and target policy.
    Q = defaultdict(lambda: defaultdict(int))
    C = {}
    target_policy = {}
    for s in states:
        C[s] = {}
        for a in actions[s]:
            C[s][a] = 0
    # Loop for each episode
    for _ in range(num_episodes):
        # Create behavior policy - diccionary of diccionaries
        behavior_policy = {}
        for s in states:
            n = len(actions[s])
            probs = np.random.dirichlet(np.ones(n))
            behavior_policy[s] = {a: p for a, p in zip(actions[s], probs)}
        # Generate en episode
        episode = generate_episode_fn_behavior_policy(behavior_policy)
        # Initialize G and W
        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            action, state, reward = episode[t]
            # Update all values
            G = gamma * G + reward
            C[state][action] = C[state][action] + W
            Q[state][action] = Q[state][action] + (W / C[state][action]) * (G - Q[state][action])
            # Update target policy
            q_vals = Q[state]
            max_value = max(q_vals.values())
            best_actions = [a for a in q_vals if q_vals[a] == max_value]
            random_idx = np.random.randint(0, len(best_actions))
            best_action = best_actions[random_idx]
            target_policy[state] = best_action
            # If target policy and behavior policy diverges
            if action != target_policy[state]:
                break
            W = W * (1 / behavior_policy[state][action])
    return Q, target_policy