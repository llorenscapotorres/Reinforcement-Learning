import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

class RewardGenerator:
    def __init__(self, k, mean_range=(-5, 5), std=1.0, seed=None):
        """
        Initialize k Gaussin Distributions with random means.
        
        Args:
            k (int): Number of actions.
            mean_range (tuple): Range of values for the mean.
            std (float): Standard Deviation for all the distributions.
            seed (int, optional): Random seed.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.k = k
        self.std = std
        self.means = np.random.uniform(mean_range[0], mean_range[1], k)

    def get_reward(self, action_index):
        """
        Returns the reward for the selected action.
        
        Args:
            action_index (int): Index of the action.
        
        Returns:
            float: Reward.
        """
        mean = self.means[action_index]
        return np.random.normal(loc=mean, scale=self.std)
    
    def get_coefficients(self):
        return self.means, self.std

def simple_bandit(k: int, 
                  num_iter: int,
                  epsilon: float,
                  seed: int):
    """
    k: number of actions the agent can take
    num_iter: number of iterations that you want the algorithm takes
    epsilon: a number between 0 and 1 --> probability of non-taking the greedy action
    seed: Random seed
    """
    if (epsilon < 0 or epsilon > 1):
        raise ValueError("'epsilon' must be between 0 and 1!")

    # Initialize estimated value-action to zero for all actions
    Q = np.zeros(k, dtype=float)

    # Initialize number of times the ith action is selected to zero for all entries
    N = np.zeros(k, dtype=int)

    # Initialize the Reward Generator
    generator = RewardGenerator(k=k, mean_range=(-5, 5), std=1, seed=seed)

    # Initialize rewards
    rewards = []

    # We train the model with reinforcement
    for _ in range(num_iter):
        # Initialize the action for every iteration
        action = -1

        # The agent take the action
        if np.random.rand() > epsilon:
            max_val = np.max(Q)
            max_idxs = np.where(Q == max_val)[0]
            action = np.random.choice(max_idxs)
        else:
            action = np.random.choice(range(k))

        # The action gives the reward
        reward = generator.get_reward(action_index=action)

        # Update the values
        N[action] = N[action] + 1
        Q[action] = Q[action] + (1/N[action])*(reward - Q[action])

        rewards.append(reward)

    # Recover the coefficients for show a plot of the reward distributions
    means, std = generator.get_coefficients()

    # Plot distributions
    x = np.linspace(-10, 10, 500)
    fig=plt.figure(figsize=(10, 6))
    for i in range(k):
        y = norm.pdf(x, loc=means[i], scale=std)
        plt.plot(x, y, label=f'Action {i} (mean = {means[i]:.2f})')
    plt.title('Reward Distributions')
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(fig)

    return N, Q, rewards

st.markdown("# Simple k-Armed Bandit Algorithm")

st.markdown("A complete bandit algorithm using incrementally computed sample averages and epsilon-greedy action selection.")

st.markdown("The agent can take 'k' actions.")

st.markdown("The rewards came from Gaussian Distributions of exact variance but different means.")

k = st.sidebar.select_slider(
    'Select number of actions (k):',
    options=list(range(2, 21)),
    value=10
)

num_iter = st.sidebar.select_slider(
    'Select number of iterations:',
    options=list(range(100, 10001)),
    value=1001
)

epsilon = st.sidebar.slider(
    'Select epsilon:',
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01
)

seed = st.sidebar.select_slider(
    'Select random seed:',
    options=list(range(1, 999999)),
    value=27
)

N, Q, rewards = simple_bandit(k=k, num_iter=num_iter, epsilon=epsilon, seed=seed)

st.markdown('### Results:')

st.write(f"Number of times selected each action: {N}")
st.write(f"Estimated action-value: {np.round(Q, 2)}")
st.write(f"The **best action** is: {np.argmax(Q)}")

with st.expander("Plot that show the reward per iteration", expanded=True):

    # Dibujar puntos unidos por líneas
    fig = plt.figure(figsize=(20, 6))
    plt.plot(range(num_iter), rewards, 'o-', color='blue')

    # Personalización opcional
    plt.xlabel('Iterations')
    plt.ylabel('Rewards')
    plt.title('Reward per Iteration')
    plt.grid(True)
    plt.legend()

    # Mostrar
    st.pyplot(fig)