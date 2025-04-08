import numpy as np


population_size = 50
input_size = 784
hidden_size = 128
output_size = 10
mutation_rate = 0.01


# def select_parents(population, fitnesses):
#     probabilities = fitnesses / fitnesses.sum()
#     parents = []
#     for i in range(population_size // 2):
#         parent1, parent2 = np.random.choice(len(population), size=1, p=probabilities)
#         parents.append((population[parent1], population[parent2]))
#     return parents


def select_parents(population, fitnesses):
    probabilities = fitnesses / fitnesses.sum()
    parents = []
    for _ in range(population_size // 2):
        parent1, parent2 = np.random.choice(len(population), size=2, replace=False, p=probabilities)
        parents.append((population[parent1], population[parent2]))
    return parents


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
    for parent1, parent2 in parents:
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_generation.append(child)
    return next_generation


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


def fitness(weights):
    # FIX HERE !!!!!!
    return np.random.rand()


def main():
    population = initialize_population()
    fitnesses = np.array([fitness(weights) for weights in population])
    parents = select_parents(population, fitnesses)
    next_generation = create_next_generation(parents)


if __name__ == "__main__":
    main()