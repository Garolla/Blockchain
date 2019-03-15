from Photo import Photo
from Slide import Slide
import Score
from random import shuffle
outfile = open('./solution.txt', 'w')

with open('./b_lovely_landscapes.txt', 'r') as f:

    map_width = 0
    map_height = 0
    customers_n = 0
    reply_n = 0

    customers_position = []
    map = [[]]

    for index, line in enumerate(f):
        if index == 0:
            l = line.strip('\n')
            val = l.split(' ')
            map_width = val[0]
            map_height = val[1]
            customers_n = val[2]
            reply_n = val[3]
        elif index <= customers_n:
            print("customers line")
            l = line.strip('\n')
            val = l.split(' ')
            customers_position[index - customers_n] = val
        else:
            print("map")

    # here I compose slides made of vertical photos
    isOdd = True
    phs = []
    shuffle(picture_v)

    i = 0
    while i < len(picture_v) - 1:
        slide = Slide([picture_v[i],picture_v[i+1]])
        slide_all.append(slide)
        i += 2

    #shuffle(slide_all)
    print(Score.totalScore(slide_all))

    outfile.write(str(len(slide_all)) + "\n")

    for s in slide_all:
        #print(s.tags)
        if len(s.photo) == 1:
            outfile.write(str(s.photo[0].ind) + "\n")
        else:
            outfile.write(str(s.photo[0].ind) + " " + str(s.photo[1].ind) + "\n")




