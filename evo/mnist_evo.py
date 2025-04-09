import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


input_size = 784  # 28x28
hidden_size = 128
output_size = 10

population_size = 50
generations = 100
mutation_rate = 0.01


def create_network(weights):
    def network(x):
        W1, b1, W2, b2 = weights
        h = np.maximum(0, np.dot(x, W1) + b1)  # relu
        y = np.dot(h, W2) + b2
        return y
    return network


# random weights
def initialize_population():
    return [
        [
            np.random.randn(input_size, hidden_size),
            np.random.randn(hidden_size),
            np.random.randn(hidden_size, output_size),
            np.random.randn(output_size)
        ]
        for _ in range(population_size)
    ]


def fitness(weights, x_train, y_train):
    network = create_network(weights)
    predictions = np.argmax(network(x_train), axis=1)
    labels = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy


def select_parents(population, fitnesses):
    probabilities = fitnesses / fitnesses.sum()
    parents_idx = np.random.choice(range(population_size), size=population_size//2, p=probabilities)
    return [population[i] for i in parents_idx]


def crossover(parent1, parent2):
    child = []
    for p1, p2 in zip(parent1, parent2):
        mask = np.random.rand(*p1.shape) > 0.5
        child.append(np.where(mask, p1, p2))
    return child


def mutate(weights):
    for i in range(len(weights)):
        if np.random.rand() < mutation_rate:
            weights[i] += np.random.randn(*weights[i].shape) * 0.1
    return weights


def create_next_generation(parents):
    next_generation = []
    for _ in range(population_size):
        parent1, parent2 = np.random.choice(parents, size=2, replace=False)
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_generation.append(child)
    return next_generation


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)

    # one hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Initialize population
    population = initialize_population()

    for generation in range(generations):
        fitnesses = np.array([fitness(weights, x_train, y_train) for weights in population])
        print(f"Generation {generation} - Best Fitness: {fitnesses.max()}")
        
        parents = select_parents(population, fitnesses)
        population = create_next_generation(parents)

    # Evaluate best individual on test set
    best_individual = population[np.argmax(fitnesses)]
    network = create_network(best_individual)

    predictions = np.argmax(network(x_test), axis=1)
    labels = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(predictions == labels)
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
