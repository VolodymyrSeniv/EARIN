from collections import deque
import time



# open map, read line by line and form 2d list
def extract_map(file_name):

    f = open(file_name)
    mylist = []
    for line in f:
        items = line.split(',')
        mylist.append([int(item) for item in items])

    return mylist
# find starting position on map (marked as 2) and store in variable start
def find_start(maze):
    for i, row in enumerate(maze):
        for j, col in enumerate(row):
            if col == 2:
                start = (i,j)
    return start
# find end position on map (marked as 3) and store in variable end
def find_end(maze):
    for i, row in enumerate(maze):
        for j, col in enumerate(row):
            if col == 3:
                end = (i,j)
    return end

# our map

# Look to the path of your current working directory
# Or: file_path = os.path.join(working_directory, 'my_file.py')

maze = extract_map("map2.txt")
# possible moves per turn(one cycle). right,left,up,down
moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
count = 0
start = find_start(maze)
end = find_end(maze)

# Printing of maze, traverse throw 2d list, and print each element
def print_maze(maze, path=[]):
    for i, row in enumerate(maze):
        for j, col in enumerate(row):
            if (i, j) == start:
                print('S', end=' ')
            elif (i, j) == end:
                print('E', end=' ')
            elif (i, j) in path:
                print('*', end=' ')
            elif col == 1:
                print(' ', end=' ')
            else:
                print('X', end=' ')
        print()

# bfs algorithm
# takes as arguments maze itself, end and start points
def bfs(maze, start, end,count):
    # Utilizing queue principle FIFO, we use deque as queue
    queue = deque([start])
    # keeping track of visited cells
    visited = set([start])
    while queue:
        # (Utilizing queue principle FIFO)
        current = queue.popleft()
        # if we reach end stop
        if current == end:
            print('number of steps: ' + str(count))
            return True
        for move in moves:
            time.sleep(0.1)
            # next move coordinate, is current cell + move from possible moves coordinate and we get right,left,up,down per cycle
            next_move = (current[0] + move[0], current[1] + move[1])
            # if not wall or off-map and if we on tile 1(step tile) or on tile 3(end tile), add next_move to list of visited tiles and put in queue.
            if 0 <= next_move[0] < len(maze) and 0 <= next_move[1] < len(maze[0]) and next_move not in visited and maze[next_move[0]][next_move[1]] == 1 or maze[next_move[0]][next_move[1]] == 3:
                visited.add(next_move)
                queue.append(next_move)
                count += 1
                # printing maze each step with set of visited cells passed as argument
                print_maze(maze, visited)
                print("\n")
                


    return False

# dfs algorithm
# it uses a separate list 'path' to keep track of the path taken from the start to the current cell
# dfs() removes the current cell from the path list before returning False, to return the correct path when the function is called recursively on neighboring cells.
def dfs(maze, start, end,count, visited=set(), path=[]):
    # Utilizing stack principle LIFO
    visited.add(start)
    path.append(start)
    print()
    print_maze(maze, path)

    if start == end:
        print('number of steps: ' + str(count))
        return True

    for move in moves:
        time.sleep(0.1)
        next_move = (start[0] + move[0], start[1] + move[1])
        if 0 <= next_move[0] < len(maze) and 0 <= next_move[1] < len(maze[0]) and next_move not in visited and maze[next_move[0]][next_move[1]] == 1 or maze[next_move[0]][next_move[1]] == 3:
            count += 1
            if dfs(maze, next_move, end,count, visited, path):
                
                return True
    # (Utilizing stack principle LIFO)
    path.pop()
    return False

# interaction with user
while True:
    print("------------------------------------------------------------------------")
    print("choose mode, please type in(bfs,dfs or end):")
    x = input()

    if x == "bfs":
        if bfs(maze, start, end, count):
            print("Path found")
        else:
            print("Path not found")
    elif x == "dfs":
        if dfs(maze, start, end, count):
            print("Path found")
        else:
            print("Path not found")
    elif x == "end":
        break
    else:
        print("wrong input")
