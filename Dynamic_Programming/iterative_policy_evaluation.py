import streamlit as st
from PIL import Image
import numpy as np

st.markdown('# Dynamic Programming: Iterative Policy Evaluation on nxn Gridworld')

with st.expander('Instructions of the game:', expanded=True):
    st.write('The nonterminal states are S={2, 3, ..., n-1}. There are four actions possible in each state, A={up, down, right, left}, ' \
    'which deterministically cause the corresponding state transitions, except that actions that would take the agent off the grid in fact leave' \
    'the state unchanged. This is an undiscounted, episodic task. The reward is -1 on all transitions until the terminal state ({1, n}) is reached.')

    # Cargar imagen (desde archivo)
    image = Image.open('gridworld.png')
    st.image(image, caption='16x16 Gridworld', use_container_width=True)

st.markdown(
'''
#### Input:
1. Suppose the agent follows a random policy.
2. Algorithm parameter: a small threshold 'theta' > 0 determining accuracy of estimation.
3. Initialize v(s) arbitrarily, for all s in S, and V (terminal) to 0.
''')

n = st.sidebar.select_slider(
    'Select number of actions rows and columns:',
    options=list(range(3, 100)),
    value=4
)

theta = st.sidebar.slider(
    'Select threshold:',
    min_value=0.0,
    max_value=1.0,
    value=0.01,
    step=0.01
)


def policy_given_row_column(matrix: np.ndarray, 
                            row_index: int, 
                            column_index: int):

    n, m = matrix.shape

    # Direcciones posibles y sus desplazamientos
    directions = {
        'left':  (0, -1),
        'right': (0,  1),
        'up':    (-1, 0),
        'down':  (1,  0),
    }

    state_values = {}

    for direction, (dr, dc) in directions.items():
        new_r, new_c = row_index + dr, column_index + dc
        if 0 <= new_r < n and 0 <= new_c < m:
            state_values[direction] = matrix[new_r, new_c]
        else:
            state_values[direction] = float('-inf')  # Movimiento no válido

    max_value = max(state_values.values())

    # Contamos cuántas direcciones tienen el valor máximo
    best_directions = [dir for dir, val in state_values.items() if val == max_value]
    prob = 1 / len(best_directions)

    # Construimos la política como diccionario
    policy = {dir: (prob if dir in best_directions else 0.0) for dir in directions}

    return policy

def iterative_policy_evaluation(theta: float, 
                                n_gridworld: int,
                                gamma = 1):
    
    # Create a random v(s) for the gridworld --> matrix nxn --> random policy
    gridworld = np.random.rand(n_gridworld, n_gridworld)

    # Define the directions
    directions = {
        'left':  (0, -1),
        'right': (0,  1),
        'up':    (-1, 0),
        'down':  (1,  0),
    }

    # Put 0 to the terminal states
    gridworld[0, 0] = 0
    gridworld[n_gridworld - 1, n_gridworld - 1] = 0

    st.markdown('### First Policy (random):')
    st.write(gridworld)

    # Init threshold to +inf
    threshold = float('inf')

    # If the threshold is sufficient small, then stop
    num_steps = 0
    while threshold > theta:
        old_state_values = np.zeros((n_gridworld, n_gridworld))
        # Loop for each state
        for i in range(n_gridworld):
            for j in range(n_gridworld):
                # We are in the terminal state
                if ((i == 0 and j == 0) or (i == (n_gridworld - 1) and j == (n_gridworld - 1))):
                    pass
                else:
                    # Compute the policy for the state
                    policy = policy_given_row_column(gridworld, i, j)
                    state_value = 0
                    for direction, prob in policy.items():
                        idx, jdx = directions[direction]
                        if prob != 0:
                            state_value += prob * (-1 + gamma * gridworld[i + idx, j + jdx])
                    # Update state value, but saving the old values to compute the threshold
                    old_state_values[i , j] = gridworld[i, j]
                    gridworld[i, j] = state_value
        num_steps += 1
        st.markdown(f'### Step number {num_steps}')
        st.write(gridworld)
        # Computing the threshold
        threshold = np.max(np.abs(old_state_values - gridworld))
    
    return gridworld

results = iterative_policy_evaluation(theta=theta, n_gridworld=n)

st.markdown('### Final Policy:')
st.write(results)