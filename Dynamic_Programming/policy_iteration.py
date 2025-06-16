import numpy as np

def policy_iteration(states, 
                     actions, 
                     transition_function, 
                     gamma=1.0, 
                     theta=1e-6):
    # 1. Initialization
    V = {s: 0.0 for s in states}
    policy = {s: np.random.choice(actions[s]) for s in states if actions[s]}
    is_terminal = {s: not actions[s] for s in states} # Estados sin acciones son terminales

    eval_iterations = 0
    while True:
        # 2. Policy Evaluation
        while True:
            delta = 0
            for s in states:
                if is_terminal[s]:
                    continue
                v = V[s]
                a = policy[s]
                V[s] = sum(
                    prob * (r + gamma * V[s_next])
                    for (s_next, r, prob) in transition_function(s, a)
                )
                delta = max(delta, abs(v - V[s]))
                print(f'Evaluating policy... delta_{delta}')
            if delta < theta:
                break
        eval_iterations += 1
        print(f'Policy evaluation converged in {eval_iterations} iterations.')
        
        # 3. Policy Improvement
        policy_stable = True
        for s in states:
            if is_terminal[s]:
                continue
            old_action = policy[s]
            action_values = {}
            for a in actions[s]:
                action_values[a] = sum(
                    prob * (r + gamma + V[s_next])
                    for (s_next, r, prob) in transition_function(s, a)
                )
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V

# Configuración del grid
n = 8 # Tamaño del grid n x n
states = [(i, j) for i in range(n) for j in range(n)]
terminal_states = [(0, 0), (n-1, n-1)]

# Acciones disponibles
ACTIONS = ['up', 'down', 'left', 'right']
actions = {
    s: [] if s in terminal_states else ACTIONS
    for s in states
}

# Movimiento según acción
def move(s, a):
    i, j = s
    if a == 'up':
        return (max(i - 1, 0), j)
    elif a == 'down':
        return (min(i + 1, n - 1), j)
    elif a == 'left':
        return (i, max(j - 1, 0))
    elif a == 'right':
        return (i, min(j + 1, n - 1))
    return s

# Función de transición determinista
def transition_function(s, a):
    s_next = move(s, a)
    r = -1 # Siempre da -1
    prob = 1.0 # Determinista
    return [(s_next, r, prob)]

# Ejecutar
policy, V = policy_iteration(states, actions, transition_function, gamma=0.9999, theta=0.001)

# Mostrar resultados
print('Política Óptima:')
for i in range(n):
    row = ''
    for j in range(n):
        s = (i, j)
        row += f'{policy.get(s, "T"):^7}' # 'T' si es terminal
    print(row)

print('\nValores:')
for i in range(n):
    row = ''
    for j in range(n):
        s = (i, j)
        row += f'{V[s]:6.2f}'
    print(row)