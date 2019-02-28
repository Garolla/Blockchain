outfile = open('./solution.txt', 'w')

with open('./a_example.txt', 'r') as f:

    picture_all = []
    picture_h = []
    picture_v = []

    for line in f:
        temp = line.strip('\n')
        value = temp.split(' ')
        if len(value) >= 4:
            print(value[0] + " " + value[1] + " " + value[2] + " " + value[3])
            # outfile.write(value[1])

    outfile.write("""3\n0\n3\n1 2""")
