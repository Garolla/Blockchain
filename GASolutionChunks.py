# https://www.youtube.com/watch?v=uCXm6avugCo


# creare una popolazione di soluzioni e selezionare le migliori(scartando le peggiori)
# da portare avanti per ogni iterazione in modo da esplorare questo set
# di soluzioni in maniera greedy
# le soluzioni migliori sono combinate tra loro per generare la prossima generazione

from Photo import Photo
from Slide import Slide
import Score as sc

import datetime
# import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
# un tool grafico per visualizzare l'andamento del genetic man mano che va avanti


def parse_file():
    with open('./e_shiny_selfies.txt', 'r') as f:
        slide_index = 0
        slide_all = []
        picture_h = []
        picture_v = []
        solution = []
        i = -1
        for line in f:
            temp = line.strip('\n')
            value = temp.split(' ')
            if len(value) >= 2:
                orientation = value[0]
                num_of_tags = value[1]
                tags = value[2:len(value)]
                # print(orientation + " " + num_of_tags)
                # print(tags)
                if orientation == "H":
                    photo = Photo(False, tags, i)
                    # print(i)
                    # photo.isVertical = False
                    # photo.tags = tags
                    picture_h.append(photo)
                    # solution.append(i)
                else:
                    photo = Photo(True, tags, i)
                    # print(i)
                    # photo.isVertical = True
                    # photo.tags = tags
                    picture_v.append(photo)

            i = i + 1

        for p in picture_h:
            s = Slide([p])
            s.ident = slide_index
            slide_index = slide_index + 1
            slide_all.append(s)

        # here I compose slides made of vertical photos
        # shuffle(picture_v)

        i = 0
        while i < len(picture_v) - 1:
            slide = Slide([picture_v[i], picture_v[i + 1]])
            slide.ident = slide_index
            slide_index = slide_index + 1
            slide_all.append(slide)
            i += 2
    return slide_all


def doGeneticsForChunk(size_of_map, number_of_groups, slide_all, index_of_chunk, number_of_chunks,
                       number_of_ga_iterations, number_of_ga_couples, crossover_ga_cut_len, crossover_ga_number_of_cuts, ga_mutation_probability, ga_number_of_winners_to_keep):

    # initialize the map and save it
    print("I am calculating scores map...\n")
    tuple = initialize_complex_map(size_of_map, number_of_groups, slide_all, index_of_chunk, number_of_chunks)
    # def initialize_reduced_complex_map(N, groups, data_set, dataset_reduction_factor, score_map_dim_factor):
    print("I did calculate scores map.\n")
    the_map = tuple[0]
    chunk_population = tuple[1]

    print("Map initialized")

    # create the starting population
    print("I am creating starting population..\n")
    population = create_starting_population(len(chunk_population), the_map, chunk_population)
    print("Starting population created")

    # last_distance = 1000000000
    last_bscore = 0
    # for a large number of iterations do:

    for i in range(0, number_of_ga_iterations):
        new_population = []

        # evaluate the fitness of the current population
        scores = score_population(population, the_map)

        # best = population[np.argmin(scores)]
        best = population[np.argmax(scores)]
        # number_of_moves = len(best)
        curr_bscore = fitness(best, the_map)
        print('NEW Iteration %i: best score is %i' % (i, curr_bscore))
        if curr_bscore != last_bscore:
            print('Iteration %i: Best so far is for a score of %i' % (i, curr_bscore))
            # plot_best(the_map, best, i)

        # allow members of the population to breed based on their relative score;
        # i.e., if their score is higher they're more likely to breed
        for j in range(0, number_of_ga_couples):
            new_1, new_2 = crossover(population[pick_mate(scores)], population[pick_mate(scores)], crossover_ga_cut_len,
                                     crossover_ga_number_of_cuts)
            new_population = new_population + [new_1, new_2]

        # mutate
        for j in range(0, len(new_population)):
            new_population[j] = np.copy(mutate(new_population[j], ga_mutation_probability, the_map))

        # keep members of previous generation
        new_population += [population[np.argmax(scores)]]
        for j in range(1, ga_number_of_winners_to_keep):
            keeper = pick_mate(scores)
            new_population += [population[keeper]]

        # add new random members
        while len(new_population) < len(slide_all):
            new_population += [create_new_member(the_map)]

        # replace the old population with a real copy
        population = list(copy.deepcopy(new_population))

        last_bscore = curr_bscore

    return best


def main():

    now = datetime.datetime.now()

    print("Current date and time using str method of datetime object:")
    print(str(now))

    # parse file
    slide_all = parse_file()

    print("File parsed")

    # parameters
    # sparseness_of_map = 0.95
    size_of_map = len(slide_all)
    population_size = 5 #10#30
    number_of_iterations = 2 #20#1000
    number_of_couples = 9
    number_of_winners_to_keep = 2
    mutation_probability = 0.05
    number_of_groups = 1
    dataset_reduction_factor = score_map_reduction_factor = 20
    crossover_cut_len = 10
    crossover_number_of_cuts = 10

    number_of_chunks = 100
    merged_data = []

    for ichunk in range(0, number_of_chunks):
        chunk_best = doGeneticsForChunk(size_of_map, number_of_groups, slide_all, ichunk, number_of_chunks,
                           number_of_iterations, number_of_couples, crossover_cut_len,
                           crossover_number_of_cuts, mutation_probability, number_of_winners_to_keep)
        merged_data = list(copy.deepcopy(merged_data)) + list(copy.deepcopy(chunk_best))

    # add to best found solution rest of data not analyzed for computation purposes
    #merged_data = list(copy.deepcopy(best)) + list(copy.deepcopy(rest_of_reduced_population))
    total_points = sc.totalScore(merged_data)
    print("FINAL SCORE:")
    print(total_points)
    print("algorithm ended")
    # print_output_file(merged_data)

    now = datetime.datetime.now()
    print("END date and time using str method of datetime object:")
    print(str(now))


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


def initialize_complex_map(N, groups, data_set, ith_chunk_nr, number_of_chunks):

    desired_number_of_elements = N // number_of_chunks
    chunk_start_index = desired_number_of_elements * ith_chunk_nr
    chunk_end_index = min(N, chunk_start_index + desired_number_of_elements)

    the_map = np.zeros((N, N))

    data_set_with_line_scores = []

    for i in range(chunk_start_index, chunk_end_index):
        line_score = 0
        for j in range(chunk_start_index, i):
            slide_first = data_set[i]
            slide_second = data_set[j]
            the_map[i][j] = sc.score(slide_first, slide_second)
            the_map[j][i] = the_map[i][j]
            line_score = line_score + the_map[i][j]
        data_set_with_line_scores.append((line_score, data_set[i]))
    # ax = sns.heatmap(the_map)

    # plt.show()
    data_set_with_line_scores.sort(key=lambda tup: tup[0])  # sorts in place

    return the_map


def create_starting_population(size, the_map, data_set):
    # this just creates a population of different routes of a fixed size.  Pretty straightforward.

    population = []

    for i in range(0, size):
        population.append(create_new_member(the_map, data_set))

    return population


def fitness_old(solution, the_map): # calculate all the distances
    # calcolo lo score totale della soluzione. Attenzione! nel esempio usa la distanza da minimizzare,
    # mentre nel nostro caso è il punteggio da massimizzare

    score = 0

    for i in range(1, len(solution)):
        # if (the_map[solution[i - 1]][solution[i]] == 0) and i != len(the_map) - 1:
        #    print("WARNING: INVALID ROUTE")
        #    print(route)
        #    print(the_map)

        slide1Id = solution[i - 1].ident
        slide2Id = solution[i].ident
        score = score + the_map[slide1Id][slide2Id]

    return score


def fitness(solution, the_map): # calculate all the distances
    # calcolo lo score totale della soluzione. Attenzione! nel esempio usa la distanza da minimizzare,
    # mentre nel nostro caso è il punteggio da massimizzare

    score = 0

    for i in range(1, len(solution)):
        #if (the_map[solution[i - 1]][solution[i]] == 0) and i != len(the_map) - 1:
        #    print("WARNING: INVALID ROUTE")
        #    print(route)
        #    print(the_map)

        #   slide1Id = solution[i - 1].ident
        #   slide2Id = solution[i].ident

        score = score + sc.score(solution[i - 1], solution[i])

    return score


def crossover(father_a, father_b, crossover_cut_len, crossover_number_of_cuts):  # merge two solutions to generate a two new ones
    # anche qui si devono tagliare dei pezzi random e swapparli le due soluzioni in esame
    # attenzione anche qui bisogna fare attenzione a generare soluzioni ammissibili
    # oltre a soluzioni provenienti dal crossover in ogni generazione vengono aggiunti nuove soluzioni generate random
    # in aggiunta alcuni membri non vengono affatto sostituiti ma portati avanti nelle prossime generazioni
    # I initially made an error here by allowing routes to crossover at any point, which obviously won't work
    # you have to insure that when the two routes cross over that the resulting routes produce a valid route
    # which means that crossover points have to be at the same position value on the map
    numberOfSlices = crossover_number_of_cuts
    cutMeasure = crossover_cut_len
    a = list(copy.deepcopy(father_a))
    b = list(copy.deepcopy(father_b))

    for i in range(numberOfSlices):
        # get a random element
        aCutIndex = random.randint(0, len(a)-1)
        aCutBorderline = min(aCutIndex+cutMeasure, len(a))
        for tbc in range(aCutIndex,aCutBorderline):
            firstSlide = a[tbc]
            # find element gotten before within other array
            bCutIndex = b.index(firstSlide) #np.where( b == firstSlide ) #
            # bCutIndex = np.where(b == firstSlide)
            # swap position within the same array
            a.insert(bCutIndex, a.pop(tbc))
            b.insert(tbc, b.pop(bCutIndex))

    return (a, b)


def mutate(route, probability, the_map): # introduces random noise to the process # forse per evitare di incorrere in minimi locali
    # la mutation ha una certa probabilità di verificarsi (numero random generato, se supera la soglia -> mutation, lui mette il 25%
    new_route = list(copy.deepcopy(route))

    for i in range(1, len(new_route)):
        if random.random() < probability:
            randIndex1 = random.randint(0, len(new_route)-1)
            randIndex2 = random.randint(0, len(new_route)-1)
            new_route.insert(randIndex1, new_route.pop(randIndex2))


    return new_route


def create_new_member(the_map, data_set): # create a random member of the population in order to start the process
    # quando crei il membro occhio che ogni soluzione tra quelle proposte sia una soluzione corretta per il problema
    # fare i controlli alla creazione del membro qui

    # here we are going to create a new route
    # the new route can have any number of steps, so we'll select that randomly
    # the structure of the route will be a vector of integers where each value is the next step in the route
    # Everyone starts at 0, so the first value in the vector will indicate where to attempt to go next.
    # That is, if v_i = 4, then that would correspond to X_0,4 in the map that was created at initialization

    # N is the size of the map, so we need to make sure that
    # we don't generate any values that exceed the size of the map

    new_pop = list(copy.deepcopy(data_set))
    random.shuffle(new_pop)
    return new_pop


def score_population(population, the_map):
    scores = []

    for i in range(0, len(population)):
        scores += [fitness(population[i], the_map)]

    return scores


def pick_mate(scores): #seleziona 2 membri alla volta (per effettuare il crossover) perchè insieme generino nuovi elemento della nuova generazione
    # le due soluzioni che vengono scelte per generarne di nuove sono scelte sulla base dei loro punteggi
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


def print_output_file(slide_all):
    outfile = open('./solution.txt', 'w')
    outfile.write(str(len(slide_all)) + "\n")

    for s in slide_all:
        # print(s.tags)
        if len(s.photo) == 1:
            outfile.write(str(s.photo[0].ind) + "\n")
        else:
            outfile.write(str(s.photo[0].ind) + " " + str(s.photo[1].ind) + "\n")


def plot_best(the_map, route, iteration_number):
    ax = sns.heatmap(the_map)

    x = [0.5] + [x + 0.5 for x in route[0:len(route) - 1]] + [len(the_map) - 0.5]
    y = [0.5] + [x + 0.5 for x in route[1:len(route)]] + [len(the_map) - 0.5]

    plt.plot(x, y, marker='o', linewidth=4, markersize=12, linestyle="-", color='white')
    plt.savefig('images/new1000plot_%i.png' % (iteration_number), dpi=300)
    plt.show()


main()