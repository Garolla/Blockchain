
from random import shuffle
import collections
outfile = open('./solution.txt', 'w')

terrain = {
    '#':-1,
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
                l = line.strip('\r\n')
                val = l.split(' ')
                map_width = int(val[0])
                map_height = int(val[1])
                customers_n = int(val[2])
                reply_n = int(val[3])
            elif index < (customers_n + 1):
                l = line.strip('\r\n')
                val = l.split(' ')
                customers_position.append((int(val[1]), int(val[0]), int(val[2])))
            else:
                l = line.strip('\r\n')
                terrain_line = []
                for c in l:
                    terrain_line.append(terrain[c])
                terrain_map.append(terrain_line)

    return map_width, map_height, reply_n, customers_position, terrain_map

def main():
    input = parse_file()
    print("INPUT:")
    print(input)

    map_width = input[0]
    map_height = input[1]
    reply_n = input[2]
    customers_pos = input[3]
    terrain = input[4]

    remaining_point = reply_n
    print(remaining_point)
    reply_pos = []

    customers_pos.sort(key=lambda tup: tup[2])
    customers_pos.reverse()
    print(customers_pos)

    for customer in customers_pos:
        remaining_point -= 1
        if remaining_point >= 0:
            if customer[1]+1 < 50:
                if terrain[customer[0]][customer[1]+1] > 0:
                    reply_pos.append((customer[0], customer[1]+1))
                    continue
            if customer[0] + 1 < 50:
                if terrain[customer[0]+1][customer[1]] > 0:
                    reply_pos.append(terrain[customer[0]+1][customer[1]])
                    continue
            if customer[0] - 1 > 0:
                if terrain[customer[0]][customer[1]-1] > 0:
                    reply_pos.append((customer[0], customer[1]-1))
                    continue
            if customer[0] - 1 > 0:
                if terrain[customer[0]-1][customer[1]] > 0:
                    reply_pos.append((customer[0]-1, customer[1]))
                    continue
        else:
            break

    print(terrain)
    print(reply_pos)
    for p in reply_pos:
        print(terrain[p[0]][p[1]])



    calculateScore(reply_pos, terrain, customers_pos, 50, 50)






def calculateScore(startArray,grid,goals,w,h) :

    outfile = open('./solution.txt', 'w')

    for goal in goals:
        for start in startArray:
            path = bfs(grid,start,goal,w,h)
            print(path)
            line = generateOutputFromSteps(start,path,len(path))
            outfile.write(str(line) + "\n")




def bfs(grid, start, goal, width,height):
    queue = collections.deque([[start]])
    seen = set([start])
    goalX = goal[0]
    goalY = goal[1]
    # print(queue)
    #print(seen)
    #print(goalX)
    #print (goalY)
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if y == goalY and x == goalX:
            print(path)
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[x2][y2] > 0 and (x2, y2) not in seen:
               # print("QQQQ")
               # print(queue)
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def generateOutputFromSteps(startCoors, steps, stepNr):
    stepString = str(startCoors[0]) + " " + str(startCoors[1]) + " "
    lastStepCoors = startCoors
    for i in range(0,stepNr):
        currCoors = steps[i]
        if lastStepCoors[0] == currCoors[0]:
            if lastStepCoors[1] < currCoors[1]:
                stepString = stepString + "R"
            else:
                stepString = stepString + "L"
        else:
            if lastStepCoors[0] < currCoors[0]:
                stepString = stepString + "D"
            else:
                stepString = stepString + "U"

    print("steps = " + stepString)
    return stepString







main()




