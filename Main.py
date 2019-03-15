from Photo import Photo
from Slide import Slide
import Score
from random import shuffle
outfile = open('./solution.txt', 'w')

with open('./b_lovely_landscapes.txt', 'r') as f:

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
            #print(orientation + " " + num_of_tags)
            #print(tags)
            if orientation == "H":
                photo = Photo(False, tags, i)
                #print(i)
                #photo.isVertical = False
                #photo.tags = tags
                picture_h.append(photo)
                #solution.append(i)
            else:
                photo = Photo(True, tags, i)
                #print(i)
                #photo.isVertical = True
                #photo.tags = tags
                picture_v.append(photo)

        i = i + 1

    for p in picture_h:
        slide_all.append(Slide([p]))

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




