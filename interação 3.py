import numpy as np

def step_function(u):
    if u >= 0:
        return 1
    else:
        return -1

def find_output(data, weights):
    u = np.dot(data, weights)
    return step_function(u)

def perceptron_train(inputs, outputs, learning_rate=0.1, max_iterations=1000):
    inputs = np.insert(inputs, 2, -1, axis=1)
    weights = np.random.rand(3)
    
    iteration = 0
    while iteration < max_iterations:
        error = 0
        for i in range(len(inputs)):
            predicted_output = find_output(inputs[i], weights)
            error += ((outputs[i] - predicted_output) ** 2) / 2
            learning_signal = learning_rate * (outputs[i] - predicted_output)
            weights += learning_signal * inputs[i]

        iteration += 1
        if error == 0:
            break
    
    return weights

def test_perceptron(weights, test_inputs):
    output = find_output(test_inputs, weights)
    return output

# Conjunto de treinamento para a porta lógica AND
and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs = np.array([-1, -1, -1, 1])

# Conjunto de treinamento para a porta lógica OR
or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_outputs = np.array([-1, 1, 1, 1])

# Treinamento do perceptron para a porta lógica AND
weights_and = perceptron_train(and_inputs, and_outputs)

# Teste do perceptron treinado para a porta lógica AND
print("AND Port:")
for i in range(len(and_inputs)):
    print(test_perceptron(weights_and, np.insert(and_inputs[i], 2, -1)), " - Saída esperada:", and_outputs[i])

# Treinamento do perceptron para a porta lógica OR
weights_or = perceptron_train(or_inputs, or_outputs)

# Teste do perceptron treinado para a porta lógica OR
print("\nOR Port:")
for i in range(len(or_inputs)):
    print(test_perceptron(weights_or, np.insert(or_inputs[i], 2, -1)), " - Saída esperada:", or_outputs[i])

# Conjunto de treinamento para a porta lógica XOR
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([-1, 1, 1, -1])

# Treinamento do perceptron para a porta lógica XOR
weights_xor = perceptron_train(xor_inputs, xor_outputs)

# Teste do perceptron treinado para a porta lógica XOR
print("\nXOR Port:")
for i in range(len(xor_inputs)):
    print(test_perceptron(weights_xor, np.insert(xor_inputs[i], 2, -1)), " - Saída esperada:", xor_outputs[i])




