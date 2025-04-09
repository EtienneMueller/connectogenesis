import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def initialize_population(pop_size):
    return [create_model() for _ in range(pop_size)]


def evaluate_population(population, x_train, y_train, x_test, y_test):
    fitness_scores = []
    print("population length", len(population))
    for model in population:
        print(model)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        fitness_scores.append(accuracy)
    return fitness_scores


def select_parents(population, fitness_scores):
    print("select parents")
    parents = np.random.choice(population, size=len(population), p=fitness_scores/np.sum(fitness_scores))
    return parents


def crossover(parent1, parent2):
    print("crossover")
    child = create_model()
    for i in range(len(child.get_weights())):
        if i % 2 == 0:
            child.get_weights()[i] = parent1.get_weights()[i]
        else:
            child.get_weights()[i] = parent2.get_weights()[i]
    return child


def mutate(child, mutation_rate=0.01):
    print("mutate")
    weights = child.get_weights()
    for i in range(len(weights)):
        if np.random.rand() < mutation_rate:
            weights[i] += np.random.normal(0, 0.1, weights[i].shape)
    child.set_weights(weights)
    return child


def genetic_algorithm(x_train,
                      y_train, 
                      x_test, 
                      y_test, 
                      pop_size=10, 
                      generations=10, 
                      mutation_rate=0.01):
    population = initialize_population(pop_size)
    for generation in range(generations):
        print("calc...")
        fitness_scores = evaluate_population(population, x_train, y_train, x_test, y_test)
        print(f'Generation {generation}, Best Fitness: {max(fitness_scores)}')
        parents = select_parents(population, fitness_scores)
        new_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    return population


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # run
    final_population = genetic_algorithm(x_train, y_train, x_test, y_test)

    final_fitness_scores = evaluate_population(
        final_population, 
        x_train, 
        y_train, 
        x_test, 
        y_test
    )
    best_model = final_population[np.argmax(final_fitness_scores)]
    print(f'Best Model Accuracy: {max(final_fitness_scores)}')

    # best_model.save('best_mnist_model.h5')


if __name__ == "__main__":
    main()
