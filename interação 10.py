import numpy as np
import matplotlib.pyplot as plt

def f(u):
    if u >= 0:
        return 1
    else:
        return -1

def findOutput(data, w):
    u = 0.0
    for i in range(len(data)):
        u += w[i] * data[i]
    return f(u)

# Inicialização
p = np.array([[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]])  # Conjunto de valores de entrada ampliados com a entrada dummy
d = np.array([1, 1, 1, -1])  # Saídas desejadas
w = np.random.rand(len(p[0]))  # Inicialização randomica dos pesos

c = 0.5  # Taxa de aprendizado
d_error = 0.01  # Erro desejado

# Lista para armazenar os erros em cada iteração
errors = []

# Treinamento do Perceptron
iter_count = 0
max_iterations = 1000  # Limite de iterações para evitar loop infinito

while iter_count < max_iterations:
    error = 0
    for i in range(len(p)):
        o = findOutput(p[i], w)
        error += ((d[i] - o) ** 2) / 2
        learning_signal = c * (d[i] - o)
        for k in range(len(p[i])):
            w[k] += learning_signal * p[i][k]

    errors.append(error)

    # Imprimir o erro após cada iteração
    print(f"Iteração {iter_count + 1}: Erro = {error}")

    if error < d_error:
        print('Convergiu em', iter_count, 'iterações')
        break

    iter_count += 1

# Plotar o gráfico de erro em relação às iterações
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel('Iterações')
plt.ylabel('Erro')
plt.title('Erro em relação às iterações')
plt.show()

# Testar o perceptron após treinamento
print(findOutput([1, 1, -1], w))
print(findOutput([1, -1, -1], w))
print(findOutput([-1, 1, -1], w))
print(findOutput([-1, -1, -1], w))
