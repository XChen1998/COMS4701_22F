from __future__ import division
from __future__ import print_function

import sys
import math
import time
import heapq
import resource
from queue import Queue
from queue import LifoQueue
global start_time, end_time

# This is the FINAL verison for COMS 4701 HM 1

#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []
        # We will have to calculate the total cost lastly, otherwise the config may not have been assigned
        self.total_cost = calculate_total_cost(self)

        # Get the index and (row, col) of empty block
        # self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        
       
        nextState = PuzzleState(list(self.config), self.n, self, 'Up', self.cost + 1)
        """
        We do  not want to consider the top three squares
        """
        possibleList = [3, 4, 5, 6, 7, 8]
        
        for i in possibleList:
            if nextState.config[i] == 0:
                
                # Move up
                nextState.config[i], nextState.config[i - 3] = nextState.config[i - 3], nextState.config[i]
                return nextState
            
        return None
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        

        nextState = PuzzleState(list(self.config), self.n, self, 'Down', self.cost + 1)
        """
        We do  not want to consider the bottom three squares
        """
        possibleList = [0, 1, 2, 3, 4, 5]
        
        for i in possibleList:
            if nextState.config[i] == 0:
                # Move down
                nextState.config[i], nextState.config[i + 3] = nextState.config[i + 3], nextState.config[i]
                return nextState
            
        return None
        
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        
        
        nextState = PuzzleState(list(self.config), self.n, self, 'Left', self.cost + 1)
        """
        We do  not want to consider the left three squares
        """
        possibleList = [1, 2, 4, 5, 7, 8]
        
        for i in possibleList:
            if nextState.config[i] == 0:
                # Move left
                nextState.config[i], nextState.config[i - 1] = nextState.config[i - 1], nextState.config[i]
                return nextState
            
        return None
        
        

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        
        
        nextState = PuzzleState(list(self.config), self.n, self, 'Right', self.cost + 1)
        """
        We do  not want to consider the right three squares
        """
        possibleList = [0, 1, 3, 4, 6, 7]
        
        for i in possibleList:
            if nextState.config[i] == 0:
                # Move right
                nextState.config[i], nextState.config[i + 1] = nextState.config[i + 1], nextState.config[i]
                return nextState
            
        return None

      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded_node
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children
    
    
    def __lt__(self, other):
        
        if (self.total_cost == other.total_cost):
            return self.cost < other.cost
        else:
            return self.total_cost < other.total_cost
    
    def __eq__(self, other):
        for i in range(self.n * self.n):
            if self.config[i] != other.config[i]:
                return False
        return True

    def __hash__(self):
        sum = 0
        for i in (self.config):
            sum = sum * 10 + i 
        
        return hash(sum)
    
    
# Function that Writes to output.txt

def findPath(final_state):
    
    state = final_state
    path = []
    while (state is not None):
        path.append(state.action)
        state = state.parent
    
    path.reverse()
    path.pop(0)
    return path


### Students need to change the method to have the corresponding parameters
def writeOutput(final, depth, max_depth, expanded_node, ram, r_time):
    ### Student Code Goes here
    path = findPath(final)
    
    file = open("output.txt", "w")
    file.write("path_to_goal: " + str(path) + "\n")
    file.write("cost_of_path: " + str(final.cost) + "\n")
    file.write("nodes_expanded: " + str(expanded_node) + "\n")
    file.write("search_depth: " + str(depth) + "\n")
    file.write("max_search_depth: " + str(max_depth) + "\n")
    file.write("running_time: " + str(format(r_time, '.8f')) + "\n")
    file.write("max_ram_usage: " + str(format(ram, '.8f')) + "\n")
    file.close()

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    global end_time
    # Here fringe is a queue
    fringe = Queue()
    fringe_set = set()
    explored = set()
    
    fringe.put(initial_state)
    fringe_set.add(initial_state)
    
    depth, expanded_node = 0, 0
    while not fringe.empty():
        cur_state = fringe.get()
        fringe_set.remove(cur_state)
        explored.add(cur_state)
        
        
        if (test_goal(cur_state)):
            end_time = time.time()
            end_ram = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (2**20)
            
            writeOutput(cur_state, cur_state.cost, depth, expanded_node, end_ram, end_time - start_time)
            return
        
        neighbours = cur_state.expand()
        for neighbour in neighbours:
            if neighbour not in explored and neighbour not in fringe_set:
                fringe.put(neighbour)
                fringe_set.add(neighbour)
                depth = depth if depth > neighbour.total_cost else neighbour.cost
        expanded_node = expanded_node + 1
                
        
                

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    global end_time
    # Here fringe is a stack
    fringe = LifoQueue()
    fringe_set = set()
    explored = set()
    fringe.put(initial_state)
    fringe_set.add(initial_state)
    depth, expanded_node = 0, 0
    while fringe.qsize() != 0:
        cur_state = fringe.get()
        
        fringe_set.remove(cur_state)
        explored.add(cur_state)
        if (test_goal(cur_state)):
            end_time = time.time()
            end_ram = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (2**20)
            writeOutput(cur_state, cur_state.cost, depth, expanded_node, end_ram, end_time - start_time)
            return
        
        neighbours = reversed(cur_state.expand())
        for neighbour in neighbours:
            if neighbour not in explored and neighbour not in fringe_set:
                fringe.put(neighbour)
                fringe_set.add(neighbour)
                depth = depth if depth > neighbour.total_cost else neighbour.cost
                
        expanded_node = expanded_node + 1                 


def A_star_search(initial_state):
     """A * search"""
     ### STUDENT CODE GOES HERE ###
     global end_time
     explored_node = set()
     fringe_set = set();
     fringe_dict_distance = {}
     # Here the fringe is a priority queue
     fringe = []
     depth = 0
     expanded_node = 0
     heapq.heappush(fringe, initial_state)
     fringe_set.add(initial_state)
     fringe_dict_distance[initial_state] = initial_state.total_cost
     # fringe_dict_config[tuple(initial_state.config)] = initial_state
     
     while len(fringe) != 0:
         cur_state = heapq.heappop(fringe)
         explored_node.add(cur_state)
         # if (tuple(cur_state.config) in fringe_set):
         fringe_set.remove(cur_state)
         fringe_dict_distance.pop(cur_state)
         # fringe_dict_config.pop(tuple(cur_state.config))
         if test_goal(cur_state):
             end_time = time.time()
             end_ram = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (2**20)
             writeOutput(cur_state, cur_state.cost, depth, expanded_node, end_ram, end_time - start_time)
             return
         
         cur_state.expand()
         for neighbour in cur_state.children:
            cur_cost = calculate_total_cost(neighbour)
            if neighbour not in explored_node and neighbour not in fringe_set:
                heapq.heappush(fringe, neighbour)
                fringe_set.add(neighbour)
                fringe_dict_distance[neighbour] = cur_cost
                # fringe_dict_config[tuple(neighbour.config)] = neighbour
                depth = depth if depth > neighbour.total_cost else neighbour.cost
                
            elif neighbour in fringe_set and fringe_dict_distance[neighbour] > cur_cost:
                
                fringe_dict_distance[neighbour] = cur_cost
                # Now let us delete the item
                fringe.remove(neighbour)
                
                # Let us heapify the heap  
                heapq.heapify(fringe)
                
                # Add it to the fringe again
                heapq.heappush(fringe, neighbour)
                    
         expanded_node = expanded_node + 1
     


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    total_cost = state.cost
    for i in range (state.n * state.n):
        total_cost = total_cost + calculate_manhattan_dist(i, state.config[i], state.n)
        
    return total_cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    if value == 0:
        return 0;
    x_goal, x_index = value // n, idx // n
    y_goal, y_index = value % n, idx % n
    return abs (x_index - x_goal) + abs (y_index - y_goal)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    for index, i in enumerate(puzzle_state.config):
        if index != i:
            return False
    return True

    

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    global start_time
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()