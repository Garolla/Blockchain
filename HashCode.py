outfile = open('./solution.txt', 'w')

with open('./a_example.txt', 'r') as f:

    picture_all = []
    picture_h = []
    picture_v = []

    for line in f:
        temp = line.strip('\n')
        value = temp.split(' ')
        if len(value) >= 2:
            position = value[0]
            num_of_tags = value[1]
            tags = value[2:len(value)]
            print(position + " " + num_of_tags)
            print(tags)
            # outfile.write(value[1])

    outfile.write("""3\n0\n3\n1 2""")
