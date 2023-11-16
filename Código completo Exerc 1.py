import numpy as np

def step_function(u):
    if u >= 0:
        return 1
    else:
        return -1

def find_output(data, weights):
    u = np.dot(data, weights)
    return step_function(u)

# Inicialização
inputs = np.array([[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]])
outputs_and = np.array([1, 1, 1, -1])
outputs_or = np.array([1, -1, -1, -1])

weights_and = np.random.rand(len(inputs[0]))

# Taxa de aprendizado
learning_rate = 0.1

# Erro desejado
desired_error = 0.01

# Número máximo de iterações
max_iterations = 1000

iteration = 0
while iteration < max_iterations:
    error = 0
    for i in range(len(inputs)):
        predicted_output = find_output(inputs[i], weights_and)
        error += ((outputs_and[i] - predicted_output) ** 2) / 2
        learning_signal = learning_rate * (outputs_and[i] - predicted_output)
        weights_and += learning_signal * inputs[i]

    iteration += 1
    print(error, " ## ", weights_and)
    if error < desired_error:
        print('Número de iterações:', iteration)
        break

# Teste para porta lógica AND
print("\nTeste para porta lógica AND:")
print(find_output([1, 1, -1], weights_and))  # Saída esperada: 1
print(find_output([1, -1, -1], weights_and))  # Saída esperada: 1
print(find_output([-1, 1, -1], weights_and))  # Saída esperada: 1
print(find_output([-1, -1, -1], weights_and))  # Saída esperada: -1

# Repetir o processo para a porta lógica OR
weights_or = np.random.rand(len(inputs[0]))

iteration = 0
while iteration < max_iterations:
    error = 0
    for i in range(len(inputs)):
        predicted_output = find_output(inputs[i], weights_or)
        error += ((outputs_or[i] - predicted_output) ** 2) / 2
        learning_signal = learning_rate * (outputs_or[i] - predicted_output)
        weights_or += learning_signal * inputs[i]

    iteration += 1
    print(error, " ## ", weights_or)
    if error < desired_error:
        print('Número de iterações:', iteration)
        break

# Teste para porta lógica OR
print("\nTeste para porta lógica OR:")
print(find_output([1, 1, -1], weights_or))  # Saída esperada: 1
print(find_output([1, -1, -1], weights_or))  # Saída esperada: -1
print(find_output([-1, 1, -1], weights_or))  # Saída esperada: -1
print(find_output([-1, -1, -1], weights_or))  # Saída esperada: -1
