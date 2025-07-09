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
        Optimal value function for the given policy. It will be a dictionary.
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

