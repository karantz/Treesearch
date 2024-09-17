import sys
import time
from collections import deque
import heapq
import datetime

class Node:
    def __init__(self, state, parent=None, move=None, depth=0, cost=0,tile_name=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.tile_name=tile_name

    def __lt__(self, other):
        return self.cost < other.cost

def parse_file(file_name):
    end_of_file_keywords = {"end of file", "END OF FILE", "EOF", "eof"}
    
    with open(file_name, 'r') as file:
        return [
            list(map(int, line.split()))
            for line in file
            if line.strip() and line.strip().upper() not in end_of_file_keywords
        ]
 
def search_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None

def is_goal(state, goal):
    return state == goal

def g_m(state):                             #generate nodes
    blank_i, blank_j = search_blank(state)
    moves = []
    
    if blank_i > 0:  # Move blank up
        moves.append("Up")
    if blank_i < 2:  # Move blank down
        moves.append("Down")
    if blank_j > 0:  # Move blank left
        moves.append("Left")
    if blank_j < 2:  # Move blank right
        moves.append("Right")
    
    return moves

def move_blank(state, direc):
    i, j = search_blank(state)
    n_s = [row[:] for row in state]
    if direc == "Up":
        tile_name = n_s[i-1][j]
        n_s[i][j], n_s[i-1][j] = n_s[i-1][j], n_s[i][j]
    elif direc == "Down":
        tile_name = n_s[i+1][j]
        n_s[i][j], n_s[i+1][j] = n_s[i+1][j], n_s[i][j]
    elif direc == "Left":
        tile_name = n_s[i][j-1]
        n_s[i][j], n_s[i][j-1] = n_s[i][j-1], n_s[i][j]
    elif direc == "Right":
        tile_name = n_s[i][j+1]
        n_s[i][j], n_s[i][j+1] = n_s[i][j+1], n_s[i][j]
    
    return n_s, tile_name

def trace_file(trace_msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(timestamp)
    trace_filename = f"trace-{timestamp}.txt"
    with open(trace_filename, 'w') as trace_file:
        trace_file.write(trace_msg)
        print(f"Trace dumped to {trace_filename}")
#dfs implementation
def bfs(start, goal, d_flag):
    frontier = deque([Node(start)])
    explored = set()
    
    trace_msg = ""
    max_f_s = 1
    nodes_pop = 0
    nodes_exp = 0
    nodes_gen = 1
    print(d_flag)
   
    while frontier:
        node = frontier.popleft()
        nodes_pop += 1
        
        if d_flag:
            trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
            trace_msg += f"Closed set: {list(explored)}\n"

        # Log trace information
      
        
        if is_goal(node.state, goal):
            if d_flag: 
                trace_file(trace_msg)
            return node, nodes_pop, nodes_exp, nodes_gen, max_f_s
        
        explored.add(tuple(map(tuple, node.state)))
        nodes_exp += 1
        
        for move in g_m(node.state):   # expanding nodes and keeping track of generated nodes
            C_s, tile_name = move_blank(node.state, move)
            if tuple(map(tuple, C_s)) not in explored:
                child_node = Node(C_s, node, move, node.depth + 1, node.cost ,tile_name)
                frontier.append(child_node)
                nodes_gen += 1

        max_f_s = max(max_f_s, len(frontier))
    if d_flag:
        trace_file(trace_msg)
    return None, nodes_pop, nodes_exp, nodes_gen, max_f_s

def a_star(start, goal, d_flag):
    frontier = []
    heapq.heappush(frontier, Node(start, cost=heuristic(start, goal)))
    explored = set()

    trace_msg = ""
    max_f_s = 1
    nodes_pop = 0
    nodes_exp = 0
    nodes_gen = 1
    print(d_flag)
    while frontier:
        node = heapq.heappop(frontier)
        nodes_pop += 1

        # Log trace information
        if d_flag:
            trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
            trace_msg += f"Closed set: {list(explored)}\n"
        
        if is_goal(node.state, goal):
            if d_flag:
                trace_file(trace_msg)
            return node, nodes_pop, nodes_exp, nodes_gen, max_f_s

        explored.add(tuple(map(tuple, node.state)))
        nodes_exp += 1

        for move in g_m(node.state):               #expanding nodes and keeping track of generated nodes
            C_s,tile_name = move_blank(node.state, move)
            if tuple(map(tuple, C_s)) not in explored:
                move_cost = 1 #assume the move cost is 1
                child_node = Node(C_s, node, move, node.depth + 1,  node.cost + move_cost + heuristic(C_s, goal),tile_name)
                heapq.heappush(frontier, child_node)
                nodes_gen += 1
                # a* the cost is movecost and heuristic cost
        max_f_s = max(max_f_s, len(frontier))
    if d_flag:
        trace_file(trace_msg)
    return None, nodes_pop, nodes_exp, nodes_gen, max_f_s
# implement ucs
def ucs(start, goal, d_flag):
    frontier = []
    heapq.heappush(frontier, Node(start, cost=0))
    explored = set()

    trace_msg = ""
    max_f_s = 1
    nodes_pop = 0
    nodes_exp = 0
    nodes_gen = 1

    while frontier:
        node = heapq.heappop(frontier)
        nodes_pop += 1

        # Log trace information
        if d_flag:
            trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
            trace_msg += f"Closed set: {list(explored)}\n"
            

        if is_goal(node.state, goal):          #if we get the goal state the put values in trace file 
            if d_flag:
                trace_file(trace_msg)
            return node, nodes_pop, nodes_exp, nodes_gen, max_f_s

        explored.add(tuple(map(tuple, node.state)))
        nodes_exp += 1

        for move in g_m(node.state):         #expanding nodes and keeping track of generated nodes
            C_s,tile_name = move_blank(node.state, move)
            if tuple(map(tuple, C_s)) not in explored:
                move_cost = 1 #assume cost is 1
                child_node = Node(C_s, node, move, node.depth + 1, node.cost + move_cost ,tile_name=tile_name) 
                heapq.heappush(frontier, child_node)
                nodes_gen += 1
                # for ucs we need to find the move cost only
        max_f_s = max(max_f_s, len(frontier))
    if d_flag:
        trace_file(trace_msg)
    return None, nodes_pop, nodes_exp, nodes_gen, max_f_s
# dfs implementation
def dfs(start, goal, d_flag):
    frontier = [Node(start)]
    explored = set()

    trace_msg = ""
    max_f_s = 1
    nodes_pop = 0
    nodes_exp = 0
    nodes_gen = 1

    while frontier:
        node = frontier.pop()
        nodes_pop += 1

        # Log trace information
        if d_flag:
            trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
            trace_msg += f"Closed set: {list(explored)}\n"

        if is_goal(node.state, goal):
            if d_flag:
                trace_file(trace_msg)
            return node, nodes_pop, nodes_exp, nodes_gen, max_f_s

        explored.add(tuple(map(tuple, node.state)))
        nodes_exp += 1

        for move in reversed(g_m(node.state)):  # Reverse for DFS to explore left-most branch first  #expanding nodes and keeping track of generated nodes
            C_s,tile_name = move_blank(node.state, move)
            if tuple(map(tuple, C_s)) not in explored:
                child_node = Node(C_s, node, move, node.depth + 1,tile_name=tile_name)
                frontier.append(child_node)
                nodes_gen += 1

        max_f_s = max(max_f_s, len(frontier))
    if d_flag:
        trace_file(trace_msg)
    return None, nodes_pop, nodes_exp, nodes_gen, max_f_s

def dls(start, goal, limit, d_flag):
    frontier = [Node(start)]
    explored = set()

    trace_msg = ""
    max_f_s = 1
    nodes_pop = 0
    nodes_exp = 0
    nodes_gen = 1

    while frontier:
        node = frontier.pop()
        nodes_pop += 1

        # Log trace information
        if d_flag:
            trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
            trace_msg += f"Closed set: {list(explored)}\n"

        if is_goal(node.state, goal):
            if d_flag:
                trace_file(trace_msg)
            return node, nodes_pop, nodes_exp, nodes_gen, max_f_s

        if node.depth < limit:
            explored.add(tuple(map(tuple, node.state)))
            nodes_exp += 1

            for move in reversed(g_m(node.state)):  # Reverse for DFS-like behavior   #expanding nodes and keeping track of generated nodes
                C_s,tile_name = move_blank(node.state, move)
                if tuple(map(tuple, C_s)) not in explored:
                    child_node = Node(C_s, node, move, node.depth + 1,tile_name)
                    frontier.append(child_node)
                    nodes_gen += 1

            max_f_s = max(max_f_s, len(frontier))
    if d_flag:
        trace_file(trace_msg)
    return None, nodes_pop, nodes_exp, nodes_gen, max_f_s

def ids(start, goal, d_flag):
    depth = 0
    nodes_pop = nodes_exp = nodes_gen = max_f_s = 0
    trace_msg = ""

    while True:
        # Initialize frontier and explored set for each depth-limited search
        frontier = [Node(start)]
        explored = set()
        local_nodes_pop = local_nodes_exp = local_nodes_gen = 0
        local_max_f_s = 1

        while frontier:
            node = frontier.pop()
            local_nodes_pop += 1

            # Collect trace data
            if d_flag:
                trace_msg += f"Depth Limit: {depth}\n"
                trace_msg += f"Current Node: {node.state}\n"
                trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
                trace_msg += f"Closed set: {list(explored)}\n\n"

            if is_goal(node.state, goal):
                nodes_pop += local_nodes_pop
                nodes_exp += local_nodes_exp
                nodes_gen += local_nodes_gen
                max_f_s = max(max_f_s, local_max_f_s)

                if d_flag:
                    trace_file(d_flag, trace_msg)

                return node, nodes_pop, nodes_exp, nodes_gen, max_f_s

            if node.depth < depth:
                explored.add(tuple(map(tuple, node.state)))
                local_nodes_exp += 1

                for move in reversed(g_m(node.state)):  # Reverse for DFS-like behavior   #expanding nodes and keeping track of generated nodes
                    C_s,tile_name = move_blank(node.state, move)
                    if tuple(map(tuple, C_s)) not in explored:
                        child_node = Node(C_s, node, move, node.depth + 1,node.cost + 1 ,tile_name=tile_name)
                        frontier.append(child_node)
                        local_nodes_gen += 1

                local_max_f_s = max(local_max_f_s, len(frontier))

        # Update global counters after each DLS iteration
        nodes_pop += local_nodes_pop
        nodes_exp += local_nodes_exp
        nodes_gen += local_nodes_gen
        max_f_s = max(max_f_s, local_max_f_s)

        # Increase depth for the next iteration
        depth += 1


def greedy_search(start, goal, d_flag):
    frontier = []
    # Initialize with the starting node, priority based on heuristic cost
    heapq.heappush(frontier, Node(start, cost=heuristic(start, goal)))
    explored = set()

    trace_msg = ""
    max_f_s = 1
    nodes_pop = 0
    nodes_exp = 0
    nodes_gen = 1

    while frontier:
        node = heapq.heappop(frontier)
        nodes_pop += 1

        # Log trace information if requested
        if d_flag:
            trace_msg += f"Fringe: {[n.state for n in frontier]}\n"
            trace_msg += f"Closed set: {list(explored)}\n"

        # Check if the goal has been reached
        if is_goal(node.state, goal):
            if d_flag:
                trace_file(trace_msg)
            return node, nodes_pop, nodes_exp, nodes_gen, max_f_s

        # Add the current node's state to the explored set
        explored.add(tuple(map(tuple, node.state)))
        nodes_exp += 1

        # Generate children and add them to the frontier
        for move in g_m(node.state):
            C_s, tile_name = move_blank(node.state, move)
            if tuple(map(tuple, C_s)) not in explored:
                # Use heuristic value as the cost
                child_node = Node(C_s, node, move, node.depth + 1, heuristic(C_s, goal), tile_name)
                heapq.heappush(frontier, child_node)
                nodes_gen += 1

        # Track the maximum size of the fringe
        max_f_s = max(max_f_s, len(frontier))
    
    # If the search completes without finding the goal
    if d_flag:
        trace_file(trace_msg)
    return None, nodes_pop, nodes_exp, nodes_gen, max_f_s


def heuristic(state, goal):  #to find heuristic use to find manhattan ditance 
    cost = 0
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            if tile != 0:  # Skip the empty tile
                goal_i, goal_j = find_tile(goal, tile)
                cost += abs(goal_i - i) + abs(goal_j - j)
    return cost


def find_tile(state, value):   #to find which tile
    for i in range(3):
        for j in range(3):
            if state[i][j] == value:
                return i, j
    return None

def print_search_result(result):    #tp print result on terminal
    node, nodes_pop, nodes_exp, nodes_gen, max_f_s = result
    if not node:
        print("No solution found.")
        return

    # Reconstruct path
    path = []
    t_cost = 0
    while node.parent:
        path.append(f"Move ({node.tile_name}) {node.move}")
        t_cost += node.cost
        node = node.parent
    path.reverse()

    print(f"Nodes Popped: {nodes_pop}")
    print(f"Nodes Expanded: {nodes_exp}")
    print(f"Nodes Generated: {nodes_gen}")
    print(f"Max Fringe Size: {max_f_s}")
    print(f"Solution found at depth {len(path)} with cost of {t_cost}.")
    print(" -> ".join(path))
    


def main():
    if len(sys.argv) < 4:
        print("Usage: python expense_8_puzzle.py <start_file> <goal_file> <search_method> True/False limit(only dls)")
        sys.exit(1)
        # tp parse the value in command line
    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    search_method = sys.argv[3]
    d_flag = len(sys.argv) > 4 and sys.argv[4] =="True"
    #d_flag =sys.argv[4] #main comment
    

    start_state = parse_file(start_file)
    goal_state = parse_file(goal_file)

    start_time = time.time()

    if search_method == 'bfs':
        result = bfs(start_state, goal_state, d_flag)
    elif search_method == 'ucs':
        result = ucs(start_state, goal_state, d_flag)
    elif search_method == 'dfs':
        result = dfs(start_state, goal_state, d_flag)
    elif search_method == 'dls':
        limit = int(sys.argv[5])  
        print(f"lmit:{limit}")# You can change the depth limit here
        result = dls(start_state, goal_state, limit, d_flag)
    elif search_method == 'ids':
        result = ids(start_state, goal_state, d_flag)
    elif search_method == 'greedy':
        result = greedy_search(start_state, goal_state, d_flag)  # Greedy uses the heuristic only
    elif search_method == 'a*':
        result = a_star(start_state, goal_state, d_flag)  # A* uses heuristic + cost
    else:
        print(f"Unknown search method: {search_method}")
        sys.exit(1)

    print_search_result(result)

    if d_flag:
        end_time = time.time() 
        print(f"Execution time: {end_time - start_time:.2f} seconds") #this shows the execution time

if __name__ == "__main__":
    main()


