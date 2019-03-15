

def calculateScore(startArray,grid,goals,w,h) :

    outfile = open('./solution.txt', 'w')

    for goal in goals:
        for start in startArray:
            path = bfs(grid,start,goal,w,h)
            line = generateOutputFromSteps(start,path,len(path))
            outfile.write(line + "\n")




def bfs(grid, start, goal, width,height):
    queue = collections.deque([[start]])
    seen = set([start])
    goalX = goal[x]
    goalY = goal[y]
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if y == goalY & x == goalX:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] > 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def generateOutputFromSteps(startCoors, steps, stepNr):
    stepString = startCoors[0] + " " + startCoors[1] + " "
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
