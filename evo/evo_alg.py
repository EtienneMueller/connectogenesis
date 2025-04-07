import numpy as np


target = [0, 1, 0, 1, 0, 0, 1, 1, 1] 
pop_size = 50
mutation_rate = 0.01 
generations = 200


# NEEDS BETTER!!!
def fitness(individual):
    return sum([abs(target[i] - individual[i]) for i in range(len(target))])


def new_gen(genome_length, population):
    parents = population[::2] 
    children = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        if i+1 > i:
            parent2 = parents[0]
        else:
            parent2 = parents[(i+1)]
        
        cutting_point = np.random.randint(1, genome_length)  
        child = parent1[:cutting_point] + parent2[cutting_point:]
    
        mutate_child(genome_length, child)
        
        children.append(child)
    
    population = children


def mutate_child(genome_length, child):
    if np.random.uniform() < mutation_rate:
        pos_to_mutate = np.random.randint(genome_length)
        child[pos_to_mutate] = 1 - child[pos_to_mutate]


def main():
    genome_length = len(target)
    population = [np.random.randint(2, size=genome_length).tolist() for _ in range(pop_size)]

    for _ in range(generations):
        # evaluate and sort
        scores = [fitness(individual) for individual in population]
        population = [x for _, x in sorted(zip(scores, population))]

        if population[-1] == target:
            break
        
        print("Best Solution: ", population[-1], "Score: ", scores[-1])

        new_gen(genome_length, population)
        

if __name__ == "__main__":
    main()
