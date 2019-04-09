#https://www.youtube.com/watch?v=uCXm6avugCo


# creare una popolazione di soluzioni e selezionare le migliori(scartando le peggiori) da portare avanti per ogni iterazione in modo da esplorare questo set
# di soluzioni in maniera greedy
# le soluzioni migliori sono combinate tra loro per generare la prossima generazione

import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
# un tool grafico per visualizzare l'andamento del genetic man mano che va avanti


def main():
    # parameters
    sparseness_of_map = 0.95
    size_of_map = 1000
    population_size = 30
    number_of_iterations = 1000
    number_of_couples = 9
    number_of_winners_to_keep = 2
    mutation_probability = 0.05
    number_of_groups = 1

    # initialize the map and save it
    the_map = initialize_complex_map(sparseness_of_map, size_of_map, number_of_groups)

    # create the starting population
    population = create_starting_population(population_size, the_map)

    last_distance = 1000000000
    # for a large number of iterations do:

    for i in range(0, number_of_iterations):
        new_population = []

        # evaluate the fitness of the current population
        scores = score_population(population, the_map)

        best = population[np.argmin(scores)]
        number_of_moves = len(best)
        distance = fitness(best, the_map)

        if distance != last_distance:
            print('Iteration %i: Best so far is %i steps for a distance of %f' % (i, number_of_moves, distance))
            #plot_best(the_map, best, i)

        # allow members of the population to breed based on their relative score;
        # i.e., if their score is higher they're more likely to breed
        for j in range(0, number_of_couples):
            new_1, new_2 = crossover(population[pick_mate(scores)], population[pick_mate(scores)])
            new_population = new_population + [new_1, new_2]

        # mutate
        for j in range(0, len(new_population)):
            new_population[j] = np.copy(mutate(new_population[j], 0.05, the_map))

        # keep members of previous generation
        new_population += [population[np.argmin(scores)]]
        for j in range(1, number_of_winners_to_keep):
            keeper = pick_mate(scores)
            new_population += [population[keeper]]

        # add new random members
        while len(new_population) < population_size:
            new_population += [create_new_member(the_map)]

        # replace the old population with a real copy
        population = copy.deepcopy(new_population)

        last_distance = distance

    # plot the results
    #plot_best(the_map, best, number_of_iterations)
    print("algorithm ended")


def print_pop(population):
    for i in population:
        print(i)


def initialize(p_zero,N):
    # initialize the problem, fetch data, load models and so on
    # random initialization for example
    # creo una matrice di distanze NxN dove l'intersezione ij mi indica la distanza tra il nodo i e il nodo j. se uguale a 0 non esiste un percorso tra le due
    the_map = np.zeros((N, N))

    for i in range(0, N):
        for j in range(0, i):
            if random.random() > p_zero:
                the_map[i][j] = random.random()
                the_map[j][i] = the_map[i][j]

    return the_map
    # create the concept of distance between each point.


# Let's make a more complicated map that has at least 10 stops that have to be made and see what happens.


def initialize_complex_map(p_zero, N, groups):
    the_map = np.zeros((N, N))

    for i in range(0, N):
        for j in range(0, i):
            group_i = int(i / (N / groups))
            group_j = int(j / (N / groups))

            if random.random() > p_zero and abs(group_i - group_j) <= 1:
                the_map[i][j] = random.random()
                the_map[j][i] = the_map[i][j]

    ax = sns.heatmap(the_map)

    plt.show()

    return the_map



def create_starting_population(size, the_map):
    # this just creates a population of different routes of a fixed size.  Pretty straightforward.

    population = []

    for i in range(0, size):
        population.append(create_new_member(the_map))

    return population


def fitness(route, the_map): # calculate all the distances
    # calcolo lo score totale della soluzione. Attenzione! nel esempio usa la distanza da minimizzare,
    # mentre nel nostro caso è il punteggio da massimizzare

    score = 0

    for i in range(1, len(route)):
        if (the_map[route[i - 1]][route[i]] == 0) and i != len(the_map) - 1:
            print("WARNING: INVALID ROUTE")
            print(route)
            print(the_map)
        score = score + the_map[route[i - 1]][route[i]]

    return score


def crossover(a,b):  # merge two solutions to generate a two new ones
    # anche qui si devono tagliare dei pezzi random e swapparli le due soluzioni in esame
    # attenzione anche qui bisogna fare attenzione a generare soluzioni ammissibili
    # oltre a soluzioni provenienti dal crossover in ogni generazione vengono aggiunti nuove soluzioni generate random
    # in aggiunta alcuni membri non vengono affatto sostituiti ma portati avanti nelle prossime generazioni
    # I initially made an error here by allowing routes to crossover at any point, which obviously won't work
    # you have to insure that when the two routes cross over that the resulting routes produce a valid route
    # which means that crossover points have to be at the same position value on the map

    common_elements = set(a) & set(b)

    if len(common_elements) == 2:
        return (a, b)
    else:
        common_elements.remove(0)
        common_elements.remove(max(a))
        value = random.sample(common_elements, 1)

    cut_a = np.random.choice(np.where(np.isin(a, value))[0])
    cut_b = np.random.choice(np.where(np.isin(b, value))[0])

    new_a1 = copy.deepcopy(a[0:cut_a])
    new_a2 = copy.deepcopy(b[cut_b:])

    new_b1 = copy.deepcopy(b[0:cut_b])
    new_b2 = copy.deepcopy(a[cut_a:])

    new_a = np.append(new_a1, new_a2)
    new_b = np.append(new_b1, new_b2)

    return (new_a, new_b)


def mutate(route, probability, the_map): # introduces random noise to the process # forse per evitare di incorrere in minimi locali
    # la mutation ha una certa probabilità di verificarsi (numero random generato, se supera la soglia -> mutation, lui mette il 25%
    new_route = copy.deepcopy(route)

    for i in range(1, len(new_route)):
        if random.random() < probability:

            go = True

            while go:

                possible_values = np.nonzero(the_map[new_route[i - 1]])
                proposed_value = random.randint(0, len(possible_values[0]) - 1)
                route = np.append(new_route, possible_values[0][proposed_value])

                if new_route[i] == len(the_map) - 1:
                    go = False
                else:
                    i += 1

    return new_route


def create_new_member(the_map): # create a random member of the population in order to start the process
    # quando crei il membro occhio che ogni soluzione tra quelle proposte sia una soluzione corretta per il problema
    # fare i controlli alla creazione del membro qui

    # here we are going to create a new route
    # the new route can have any number of steps, so we'll select that randomly
    # the structure of the route will be a vector of integers where each value is the next step in the route
    # Everyone starts at 0, so the first value in the vector will indicate where to attempt to go next.
    # That is, if v_i = 4, then that would correspond to X_0,4 in the map that was created at initialization

    # N is the size of the map, so we need to make sure that
    # we don't generate any values that exceed the size of the map

    N = len(the_map)

    route = np.zeros(1, dtype=int)

    go = True

    i = 1

    while go:

        possible_values = np.nonzero(the_map[route[i - 1]])
        proposed_value = random.randint(0, len(possible_values[0]) - 1)
        route = np.append(route, possible_values[0][proposed_value])

        if route[i] == N - 1:
            go = False
        else:
            i += 1

    return route


def score_population(population, the_map):
    scores = []

    for i in range(0, len(population)):
        scores += [fitness(population[i], the_map)]

    return scores


def create_next_generation(population): #crea un loop con cui crea un certo numero di membri, la nuova generazione della popolazione
    return


def pick_mate(scores): #seleziona 2 membri alla volta (per effettuare il crossover) perchè insieme generino nuovi elemento della nuova generazione
    array = np.array(scores)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))

    fitness = [len(ranks) - x for x in ranks]

    cum_scores = copy.deepcopy(fitness)

    for i in range(1, len(cum_scores)):
        cum_scores[i] = fitness[i] + cum_scores[i - 1]

    probs = [x / cum_scores[-1] for x in cum_scores]

    rand = random.random()

    for i in range(0, len(probs)):
        if rand < probs[i]:
            return i


def plot_best(the_map, route, iteration_number):
    ax = sns.heatmap(the_map)

    x = [0.5] + [x + 0.5 for x in route[0:len(route) - 1]] + [len(the_map) - 0.5]
    y = [0.5] + [x + 0.5 for x in route[1:len(route)]] + [len(the_map) - 0.5]

    plt.plot(x, y, marker='o', linewidth=4, markersize=12, linestyle="-", color='white')
    plt.savefig('images/new1000plot_%i.png' % (iteration_number), dpi=300)
    plt.show()







main()