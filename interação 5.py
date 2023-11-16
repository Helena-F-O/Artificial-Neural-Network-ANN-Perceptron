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
        print("Iteration:", iteration, " - Weights:", weights)
        if error == 0:
            break
    
    return weights

def test_perceptron(weights, test_inputs):
    output = find_output(test_inputs, weights)
    return output

# Conjunto de treinamento para a porta lógica AND
and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs = np.array([-1, -1, -1, 1])

# Treinamento do perceptron para a porta lógica AND
weights_and = perceptron_train(and_inputs, and_outputs)

# Conjunto de treinamento para a porta lógica OR
or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_outputs = np.array([-1, 1, 1, 1])

# Treinamento do perceptron para a porta lógica OR
weights_or = perceptron_train(or_inputs, or_outputs)

# Conjunto de treinamento para a porta lógica XOR
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([-1, 1, 1, -1])

# Treinamento do perceptron para a porta lógica XOR
weights_xor = perceptron_train(xor_inputs, xor_outputs)
