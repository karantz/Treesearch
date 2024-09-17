UTA ID : 1002240528 
Name : Karan Arpan Thakkar
Subject : 2248-CSE-5360-005-ARTIFICIAL INTELLIGENCE I

Programming Language: Python
Version : Python 3.9.6

Code Structure:

The code is well-structured and aims to solve the 8-puzzle problem with different search algorithms. It starts with a Node class to represent a state in the search space. This class includes details like the current state, parent node, move made, depth, cost, and tile name. It also allows comparison based on cost. The parse_file() function reads puzzle states from a file and skips lines that show the file's end. Helper functions like search_blank() is_goal(), g_m(), and move_blank() do key tasks. These include finding the blank tile checking if the goal is reached, creating possible moves, and updating the state. The trace_file() function creates timestamped logs of the search process when asked. The code has several search algorithms: BFS, DFS, UCS, DLS, IDS, A*, and Greedy. Each one handles a frontier, explores nodes, and keeps track of metrics such as nodes popped, expanded, and generated. These algorithms can also log detailed trace info if needed. Heuristic functions heuristic() to calculate Manhattan distance and find_tile() to locate tile positions, help with A* and Greedy searches. The print_search_result() function shows the search results, including the solution path and stats.Finally the main() function deals with command-line arguments, sets up the search based on what the user wants, and shows results. If tracing is turned on, it also displays how long the program took to run. The code has a good structure, with separate parts that work together. It can handle different ways of searching and has solid options for displaying information.

Run command:

On Output first run the code and then write the below code on terminal


python expense_8_puzzle.py <start_file> <goal_file> <search_method> True/False [limit]

by default it will not generate trace file for true and false.
awlays use True for generating trace file.
<start_file>: Path to the file containing the start state.
<goal_file>: Path to the file containing the goal state.
<search_method>: The search algorithm to use (bfs, ucs, dfs, dls, ids, greedy, a*).
True/False: Optional flag to enable or disable tracing. If omitted, tracing is disabled.
[limit]: Only required for DLS; specifies the depth limit(only for DLS)


