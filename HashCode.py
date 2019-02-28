from random import shuffle
from collections import defaultdict

outfile = open('./solution.txt', 'w')
photos_for_tag = defaultdict(list)
with open('./b_lovely_landscapes.txt', 'r') as f:
    picture_all = []
    picture_h = []
    picture_v = []
    solution = []
    solution_temp = []

    i = -1 #to skip first line
    for line in f:
        temp = line.strip('\n')
        value = temp.split(' ')
        if len(value) >= 2:
            orientation = value[0]
            num_of_tags = value[1]
            tags = value[2:len(value)]
            #print(orientation + " " + num_of_tags)
            #print(tags)
            photo = {
                    "orientation": orientation,
                    "index": i,
                    "tags": tags
                }


            if orientation == "H":
                for t in tags:
                    photos_for_tag[t].append(i)
        i = i + 1

    for (data) in photos_for_tag.values():
        solution_temp += data

    print(photos_for_tag)
    solution = list(set(solution_temp))

    outfile.write(str(len(solution)) + "\n")

    for line in solution:
        outfile.write(str(line) + "\n" )
