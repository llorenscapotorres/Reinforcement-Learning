import numpy as np

def td_policy_evaluation_one_step(non_terminal_states: list,
                                  terminal_states: list,
                                  initial_states: list,
                                  policy: dict,
                                  next_step_fn,
                                  gamma: float = 1.0,
                                  alpha: float = 0.9,
                                  num_episodes: int = 10000):
    """
    Algorithm to evaluate a given policy, using the TD(0) method. 
    It must be an episodic task with terminal state.

    Args:
        non_terminal_states (list): A list of all possible non terminal states.
        terminal_states (list): A list of all possible terminal states.
        initial_states (list): A list of all possible initial states.
        policy (dict): The policy that will be evaluated, it can be deterministic or stochastic. It will be a dictionary of dictionaries.
        next_step_fn (function): Takes as input a state and an action, and returns a (state, reward) tuple that follows the state-action pair.
        gamma (float): Discount factor.
        alpha (float): Step size between (0, 1].
        num_episodes (int): Number of episodes used to train the model.
    Return:
        V (dict(int)): Optimal state value function for the given policy. It will be a dictionary.
    """
    # Initialize the value function arbitrarily for all states, except for V(terminal) = 0
    V = {}
    for s in non_terminal_states:
        # Value between 0 and 1
        V[s] = np.random.random()
    for s in terminal_states:
        # Value 0
        V[s] = 0
    # Loop for each episode
    for _ in range(num_episodes):
        # Initialize S
        state = np.random.choice(initial_states)
        terminal = False
        while not terminal:
            # Action given by the policy for 'state'
            actions = list(policy[state].keys())
            probs = list(policy[state].values())
            action = np.random.choice(actions, p=probs)
            # Observe the next state and reward given the 'action'
            next_state, reward = next_step_fn(state, action)
            # Update values
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
            # Until state is terminal
            if state in terminal_states:
                terminal = True
    return V        

def td_sarsa_control(non_terminal_states: list,
                    terminal_states: list,
                    initial_states: list,
                    actions: dict,
                    next_step_fn,
                    gamma: float = 1.0,
                    alpha: float = 0.1,
                    epsilon: float = None,
                    num_episodes: int = 10000,
                    initial_Q: dict = None):
    """
    Algorithm to obtain an optimal policy and Q-value function, using on-policy TD(0) control.
    To keep exploration and exploitation we will make te policy epsilon-greedy during the train. And the default value is 1/t.

    Args:
        non_terminal_states (list): A list of all possible non terminal states.
        terminal_states (list): A list of all possible terminal states.
        initial_states (list): A list of all possible initial states.
        actions (dict): A dictionary where each key represents a state and the value is a list of the possible actions that agent can take.
        next_step_fn (function): Takes as input a state and an action, and returns a (state, reward) tuple that follows the state-action pair.
        gamma (float): Discount reward's factor.
        alpha (float): Step size between (0, 1].
        epsilon (float): Probability of not taking the greedy action at once.
        num_episodes (int): Number of episodes used to train the model.
        initial_Q (dict): If it is None the values of the function are given randomly between 0 and 1. If not, it is a dictionary of dictionaries (state -> dict(actions) -> value)
    Returns:
        Q (dict(dict)): Optimal state-action pair value function. 
            It is a dictionary of dictionaries, each key representing a state (non-terminal) and each next key representing an action with their value associated.
        policy (dict): Optimal policy. Greedy with respect to Q. It gives the optimal action given the state.
    """
    # Initialize Q (state-action value function)
    if initial_Q is None:
        Q = {}
        for s in non_terminal_states:
            Q[s] = {}
            for a in actions[s]:
                Q[s][a] = np.random.random()
        for s in terminal_states:
            Q[s] = {}
            for a in actions[s]:
                Q[s][a] = 0
    else:
        Q = initial_Q
    # Initilize the policy. Greedy with respect Q
    policy = {}
    for s in non_terminal_states:
        max_Q = max(Q[s].values())
        best_actions = [a for a, q in Q[s].items() if q == max_Q]
        best_action_idx = np.random.randint(0, len(best_actions))
        policy[s] = best_actions[best_action_idx]
    # Loop for each episode
    for i in range(num_episodes):
        # Select an initial state to begin the episode
        inital_state_random_idx = np.random.randint(0, len(initial_states))
        state = initial_states[inital_state_random_idx]
        # Choose the action from the initial state following the policy (it must be epsilon-greedy).
        if epsilon == None:
            epsilon_was_none = True
            # The first action is completly random because the epsilon would be 1/t with t = 1.
            idx = np.random.randint(0, len(actions[state]))
            action = actions[state][idx]
        else:
            epsilon_was_none = False
            # If the greedy action is selected
            if np.random.random() > epsilon:
                action = policy[state]
            # If the greedy action is not selected
            else:
                idx = np.random.randint(0, len(actions[state]))
                action = actions[state][idx]
        # Loop until the episode is finished
        t = 1
        while True:
            # If epsilon is None, then the epsilon will be 1/t.
            t += 1
            if epsilon_was_none:
                epsilon = 1/t
            next_state, reward = next_step_fn(state, action)
            # Choose the next action from the state
            if np.random.random() > epsilon:
                next_action = policy[state]
            else:
                idx = np.random.randint(0, len(actions[next_state]))
                next_action = actions[next_state][idx]
            # Update Q
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            # Update the policy with greediness respect to Q
            max_Q = max(Q[state].values())
            best_actions = [a for a, q in Q[state].items() if q == max_Q]
            best_action_idx = np.random.randint(0, len(best_actions))
            policy[state] = best_actions[best_action_idx]
            # Update state and action
            state, action = next_state, next_action
            # Until state is terminal
            if state in terminal_states:
                break
    return Q, policy

