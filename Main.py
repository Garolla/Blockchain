
from random import shuffle
outfile = open('./solution.txt', 'w')

terrain = {
    '#':99999999,
    '~': 800,
    '*': 200,
    '+': 150,
    'X': 120,
    '_': 100,
    'H': 70,
    'T': 50
}

def parse_file():
    with open('./1_victoria_lake.txt', 'r') as f:
        map_width = 0
        map_height = 0
        customers_n = 0
        reply_n = 0

        customers_position = []
        customers_money = []
        terrain_map = []
        for index, line in enumerate(f):
            if index == 0:
                print("FIRST LINE")
                l = line.strip('\r\n')
                val = l.split(' ')
                map_width = int(val[0])
                map_height = int(val[1])
                customers_n = int(val[2])
                reply_n = int(val[3])
            elif index < (customers_n + 1):
                print("Customers line")
                print(index)
                l = line.strip('\r\n')
                val = l.split(' ')
                print(val)
                customers_position.append((int(val[0]), int(val[1])))
                customers_money.append(int(val[2]))
            else:
                l = line.strip('\r\n')
                terrain_line = []
                for c in l:
                    terrain_line.append(terrain[c])
                print(terrain_line)
                terrain_map.append(terrain_line)


        print(terrain_map)

    return map_width, map_height, customers_n, customers_position, customers_money, terrain_map

def main():
    input = parse_file()
    print("INPUT:")
    print(input)


main()




