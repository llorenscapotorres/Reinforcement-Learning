import numpy as np
from typing import Callable, Tuple, Any

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
                    next_step_fn: Callable[[Any, Any], Tuple[Any, float]],
                    gamma: float = 1.0,
                    alpha: float = 0.1,
                    epsilon: float = None,
                    num_episodes: int = 10000,
                    initial_Q: dict = None):
    """
    Algorithm to obtain an optimal policy and Q-value function, using on-policy TD(0) SARSA control.
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

def q_learning_control(non_terminal_states: list,
                       terminal_states: list,
                       initial_states: list,
                       actions: dict,
                       next_step_fn: Callable[[Any, Any], Tuple[Any, float]],
                       gamma: float = 1.0,
                       alpha: float = 0.1,
                       epsilon: float = None,
                       num_episodes: int = 10000,
                       initial_Q: dict = None):
    """
    Algorithm to obtain an optimal policy and Q-value function, using off-policy TD(0) Q-learning control.
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
        initial_Q (dict): If it is None the values of the function are given randomly between 0 and 1. If not, it is a dictionary of dictionaries (state -> dict(actions) -> value).
    
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
    # Loop for each episode
    for _ in range(num_episodes):
        # Select a random intial state
        idx_random_initial_state = np.random.randint(0, len(initial_states))
        state = initial_states[idx_random_initial_state]
        # Loop for each step of episode until the episode is finished
        t = 0
        while True:
            t += 1
            # Choose action from state using a policy dervied from Q that is epsilon-greedy
            if epsilon == None:
                action = take_action(state=state,
                                     Q=Q,
                                     epsilon_greedy=True,
                                     epsilon=1/t)
            else:
                action = take_action(state=state,
                                     Q=Q,
                                     epsilon_greedy=True,
                                     epsilon=epsilon)
            # Observe reward and next_state
            next_state, reward = next_step_fn(state, action)
            # Compute de max Q value for next_state
            max_action_Q = max(Q[next_state].values())
            # Update Q-learning rule
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * max_action_Q - Q[state][action])
            state = next_state
            # Until the next_state is terminal
            if next_state in terminal_states:
                break
    # Obtain policy greedily following Q
    policy = obtain_policy(Q=Q,
                           states=non_terminal_states)
    return Q, policy

def double_q_learning_control(non_terminal_states: list,
                                terminal_states: list,
                                initial_states: list,
                                actions: dict,
                                next_step_fn: Callable[[Any, Any], Tuple[Any, float]],
                                gamma: float = 1.0,
                                alpha: float = 0.1,
                                epsilon: float = None,
                                num_episodes: int = 10000,
                                initial_Q1: dict = None,
                                initial_Q2: dict = None):
    """
    Algorithm to obtain an optimal policy and Q-value function, using off-policy TD(0) Double Q-learning control.
    To keep exploration and exploitation we will make te policy epsilon-greedy during the train. And the default value of epsilon is 1/t.
        Moreover, the behavior policy will follow the value function Q1 + Q2.

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
        initial_Q1 (dict): If it is None the values of the function are given randomly between 0 and 1. If not, it is a dictionary of dictionaries (state -> dict(actions) -> value).
        initial_Q2 (dict): If it is None the values of the function are given randomly between 0 and 1. If not, it is a dictionary of dictionaries (state -> dict(actions) -> value).
    
    Returns:
        Q (dict(dict)): Optimal state-action pair value function. 
            It is a dictionary of dictionaries, each key representing a state (non-terminal) and each next key representing an action with their value associated.
        policy (dict): Optimal policy. Greedy with respect to Q. It gives the optimal action given the state.
    """
    # Initialize the first value function, Q_1
    Q1 = initial_Q1
    if initial_Q1 == None:
        Q1 = _initialize_Q(non_terminal_states=non_terminal_states,
                          terminal_states=terminal_states,
                          actions=actions)
    # Initialize the second value function, Q_2
    Q2 = initial_Q2
    if initial_Q2 == None:
        Q2 = _initialize_Q(non_terminal_states=non_terminal_states,
                          terminal_states=terminal_states,
                          actions=actions)
    # Initialize the value function Q1 + Q2
    Q_sum = {
        state: {
            action: Q1[state][action] + Q2[state][action]
            for action in Q1[state]
        }
        for state in Q1
    }
    # Loop for each episode
    for ep in range(num_episodes):
        t = 0
        # Initialize the first state
        idx_initial_state = np.random.randint(len(initial_states))
        state = initial_states[idx_initial_state]
        # Loop for each step of episode until it is terminal
        while state not in terminal_states:
            t += 1
            # Choose A from S using the policy epsilon-greedy in Q1 + Q2
            if epsilon != None:
                action = take_action(state=state,
                                     Q=Q_sum,
                                     epsilon_greedy=True,
                                     epsilon=epsilon)
            else:
                action = take_action(state=state,
                                     Q=Q_sum,
                                     epsilon_greedy=True,
                                     epsilon=1/t)
            # Observe the reward and the next state
            next_state, reward = next_step_fn(state, action)
            # With probability 0.5 update Q1 otherwise Q2
            if np.random.random() > 0.5:
                q1 = Q1[state][action]
                Q1[state][action] = Q1[state][action] + alpha * (reward + gamma * Q2[next_state][take_action(state=next_state,
                                                                                                             Q=Q1,
                                                                                                             epsilon_greedy=False)] - Q1[state][action])
                # Update Q1 + Q2 (Q_sum)
                Q_sum[state][action] += Q1[state][action] - q1
            else:
                q2 = Q2[state][action]
                Q2[state][action] = Q2[state][action] + alpha * (reward + gamma * Q1[next_state][take_action(state=next_state,
                                                                                                             Q=Q2,
                                                                                                             epsilon_greedy=False)] - Q2[state][action])
                # Update Q1 + Q2 (Q_sum)
                Q_sum[state][action] += Q2[state][action] - q2
            # Update the state
            state = next_state
    # Compute the policy
    policy = obtain_policy(Q=Q_sum,
                           states=non_terminal_states)
    # Return the policy argmaximizing Q1 + Q2, and the value function
    return policy, Q_sum

def _initialize_Q(non_terminal_states: list,
                  terminal_states: list,
                  actions: dict):
    """
    Function that initialize a Q state-action value function with all their values between [0, 1.0). 
    """
    Q = {}
    for state_list, value_fn in [(non_terminal_states, lambda: np.random.random()),
                                    (terminal_states, lambda: 0)]:
        for state in state_list:
            Q[state] = {action: value_fn() for action in actions[state]}
    return Q


def take_action(state: Any,
                Q: dict,
                epsilon_greedy: bool = True,
                epsilon: float = 0.1):
    """
    Takes action from the given state following the Q value function.
    By default, it is epsilon-greedy

    Args:
        state: Given state where the agent will take action.
        Q: The Q value function that the agent will use to take action. It is a dictionary of dictionaries.
        epsilon_greedy: A bool that indicates if the action will be epsilon greedy or not. If not is because it is greedy.
        epsilon: The epsilon value if the action is epsilon greedy.
    Returns:
        The action following the state and the "argmax Q policy".
    """
    actions = list(Q[state].keys())
    if epsilon_greedy:
        if np.random.random() > epsilon:
            q_max = max(Q[state].values())
            best_actions = [a for a, q in Q[state].items() if q == q_max]
            idx = np.random.randint(0, len(best_actions))
            return best_actions[idx]
        else:
            idx = np.random.randint(0, len(actions))
            return actions[idx]
    else:
        q_max = max(Q[state].values())
        best_actions = [a for a, q in Q[state].items() if q == q_max]
        idx = np.random.randint(0, len(best_actions))
        return best_actions[idx]
    
def obtain_policy(Q: dict,
                  states: list):
    """
    Args:
        Q: The Q optimal value function that will be used to obtain the target policy. It is a dictionary of dictionaries.
        states: The states that the policy have to take care of. Usually, only non-terminal states.
    Returns:
        policy: A dictionary where each key is a state and each valye is the action that the agent have to take in that state.
    """
    policy = {}
    for state in states:
        q_max = max(Q[state].values())
        best_actions = [a for a, q in Q[state].items() if q == q_max]
        idx = np.random.randint(0, len(best_actions))
        policy[state] = best_actions[idx]
    return policy