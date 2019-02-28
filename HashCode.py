from Photo import Photo, Slide

outfile = open('./solution.txt', 'w')

with open('./b_lovely_landscapes.txt', 'r') as f:

    picture_all = []
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
                photo = Photo()
                photo.isVertical = False
                photo.tags = tags
                picture_h.append(photo)
                solution.append(i)
            else:
                photo = Photo()
                photo.isVertical = True
                photo.tags = tags
                picture_v.append()

        i = i + 1

    outfile.write(str(len(solution)) + "\n")
    for line in solution:
        outfile.write(str(line) + "\n" )
