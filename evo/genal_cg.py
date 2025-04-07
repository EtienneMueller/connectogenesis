import random


population_size = 10
gene_length = 5  # Because 2^5 = 32 > 31
generations = 100
mutation_rate = 0.05


def fitness(x):
    return x ** 2


def create_population(size, gene_length):
    population = []
    for _ in range(size):
        individual = [random.randint(0, 1) for _ in range(gene_length)]
        population.append(individual)
    return population


def decode(individual):
    # convert binary genes to integer
    return int("".join(map(str, individual)), 2)

# select individuals if fit
def selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_index = random.choices(range(len(population)), probabilities, k=2)
    return [population[selected_index[0]], population[selected_index[1]]]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]


# here is the magic
def genetic_algorithm(pop_size, gene_length, generations, mutation_rate):
    population = create_population(pop_size, gene_length)

    for generation in range(generations):
        # Decode individuals and calculate their fitness
        fitnesses = [fitness(decode(individual)) for individual in population]
        
        # Create the next generation
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            mutate(offspring1, mutation_rate)
            mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])

        population = new_population

        # print best individual from the current generation
        best_individual = max(population, key=lambda ind: fitness(decode(ind)))
        best_fitness = fitness(decode(best_individual))
        print(f"Generation {generation}: "
              + f"Best Fitness = {best_fitness}, "
              + f"Best Individual = {decode(best_individual)}")

    return max(population, key=lambda ind: fitness(decode(ind)))

def main():
    best_solution = genetic_algorithm(population_size, gene_length, generations, mutation_rate)
    print(f"Best solution: {decode(best_solution)} with fitness {fitness(decode(best_solution))}")

if __name__ == "__main__":
    main()
