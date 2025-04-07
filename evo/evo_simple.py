import random


population_size = 10
generations = 20
mutation_rate = 0.1


def fitness_function(x):
    return -x**2 + 4*x


def create_individual():
    return random.uniform(0, 4)


def create_population(size):
    return [create_individual() for _ in range(size)]


def evaluate_population(population):
    return [fitness_function(individual) for individual in population]


def select_parents(population, fitnesses):
    parents = []
    for _ in range(len(population) // 2):
        tournament = random.sample(list(zip(population, fitnesses)), 3)
        parents.append(max(tournament, key=lambda x: x[1])[0])
    return parents


def crossover(parent1, parent2):
    return (parent1 + parent2) / 2


def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        return individual + random.uniform(-1, 1)
    return individual


def evolve_population(population, mutation_rate):
    fitnesses = evaluate_population(population)
    parents = select_parents(population, fitnesses)
    
    next_generation = []
    while len(next_generation) < len(population):
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        next_generation.append(child)
    return next_generation


def main():
    population = create_population(population_size)

    for generation in range(generations):
        population = evolve_population(population, mutation_rate)
        best_individual = max(population, key=fitness_function)
        best_fitness = fitness_function(best_individual)
        print(f"Generation {generation+1}: Best Individual = {best_individual}, Best Fitness = {best_fitness}")


if __name__ == "__main__":
    main()
