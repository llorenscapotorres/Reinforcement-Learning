import numpy as np

def value_iteration(states, actions, transition_function,
                    gamma=0.99, theta=1e-6):
    
    # 1. Initialization

    # Valor 0 para todos los states del problema
    V = {s: 0.0 for s in states}
    # Miramos las opciones de acciones que hay para cada state, y escogemos una acción aleatoria --> policy determinista
    policy = {s: np.random.choice(actions[s]) for s in states if actions[s]}
    # Estados sin acciones son terminales
    is_terminal = {s: not actions[s] for s in states}
    # Forma para comprobar que el algoritmo funciona
    eval_iterations = 0
    # Policy Evaluation
    while True:
        # Iniciamos delta
        delta = 0
        for s in states:
            if is_terminal[s]:
                continue
            # Guardamos el antiguo valor de V
            v = V[s]
            # Creamos un diccionario para almacenar los diferentes valores según la acción
            s_state_value_by_action = {action: 0 for action in actions[s]}
            # Actualizamos el diccionario
            for action in actions[s]:
                s_state_value_by_action[action] = sum(
                    prob * (r + gamma * V[s_next])
                    for (s_next, r, prob) in transition_function(s, action)
                )
            # Actualizar el valor V
            V[s] = max(s_state_value_by_action.values())
            # Calculamos la delta
            delta = max(delta, abs(v - V[s]))
        eval_iterations += 1
        print(f'Iteración número {eval_iterations}.')
        if delta < theta:
            print(f'Policy Evaluation converged in {eval_iterations} iterations.')
            break
    # Policy Improvement
    for s in states:
        # Si es terminal no se actualiza la policy porque no hay opciones
        if is_terminal[s]:
            continue
        # Guardaremos los valores de q(s, a) para cada acción posible
        action_values = {}
        for action in actions[s]:
            action_values[action] = sum(
                prob * (r + gamma * V[s_next])
                for (s_next, r, prob) in transition_function(s, action)
            )
        # Hacemos el argmax para saber cuál es la mejor opción para cada state
        best_action = max(action_values, key=action_values.get)
        # Actualizamos la policy
        policy[s] = best_action
    # Devolvemos los valores
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
policy, V = value_iteration(states, actions, transition_function, gamma=0.9999, theta=0.001)

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