import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def find_output(data, weights, activation_func=sigmoid):
    u = np.dot(data, weights)
    return activation_func(u)

def perceptron_train(inputs, outputs, learning_rate=0.01, max_iterations=5000, activation_func=sigmoid):
    inputs = np.insert(inputs, 2, -1, axis=1)
    weights = np.random.uniform(-1, 1, 3)
    
    iteration = 0
    while iteration < max_iterations:
        error = 0
        for i in range(len(inputs)):
            predicted_output = find_output(inputs[i], weights, activation_func)
            error += ((outputs[i] - predicted_output) ** 2) / 2
            learning_signal = learning_rate * (outputs[i] - predicted_output) * predicted_output * (1 - predicted_output)
            weights += learning_signal * inputs[i]

        iteration += 1
        if error == 0:
            break
    
    return weights

def test_perceptron(weights, test_inputs, activation_func=sigmoid):
    output = find_output(np.insert(test_inputs, 2, -1), weights, activation_func)
    return output

def evaluate_performance(weights, test_inputs, expected_outputs, activation_func=sigmoid):
    correct_predictions = 0
    for i in range(len(test_inputs)):
        predicted_output = test_perceptron(weights, test_inputs[i], activation_func)
        if round(predicted_output) == expected_outputs[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_inputs)
    return accuracy

# Conjunto de treinamento para a porta lógica AND
and_inputs_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs_train = np.array([-1, -1, -1, 1])

# Conjunto de teste para a porta lógica AND
and_inputs_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs_test = np.array([-1, -1, -1, 1])

# Treinamento do perceptron para a porta lógica AND
weights_and = perceptron_train(and_inputs_train, and_outputs_train)

# Avaliação de desempenho para a porta lógica AND
accuracy_and = evaluate_performance(weights_and, and_inputs_test, and_outputs_test)
print("Accuracy for AND gate:", accuracy_and)

# Conjunto de treinamento para a porta lógica OR
or_inputs_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_outputs_train = np.array([-1, 1, 1, 1])

# Conjunto de teste para a porta lógica OR
or_inputs_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_outputs_test = np.array([-1, 1, 1, 1])

# Treinamento do perceptron para a porta lógica OR
weights_or = perceptron_train(or_inputs_train, or_outputs_train)

# Avaliação de desempenho para a porta lógica OR
accuracy_or = evaluate_performance(weights_or, or_inputs_test, or_outputs_test)
print("Accuracy for OR gate:", accuracy_or)

# Conjunto de treinamento para a porta lógica XOR
xor_inputs_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs_train = np.array([-1, 1, 1, -1])

# Conjunto de teste para a porta lógica XOR
xor_inputs_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs_test = np.array([-1, 1, 1, -1])

# Treinamento do perceptron para a porta lógica XOR
weights_xor = perceptron_train(xor_inputs_train, xor_outputs_train)

# Avaliação de desempenho para a porta lógica XOR
accuracy_xor = evaluate_performance(weights_xor, xor_inputs_test, xor_outputs_test)
print("Accuracy for XOR gate:", accuracy_xor)