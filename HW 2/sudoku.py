#!/usr/bin/env python
#coding:utf-8

"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8

This code is even faster compared with the previous versions.
"""

import numpy as np
import time
import sys
import copy

ROW = "ABCDEFGHI"
COL = "123456789"

pos_str = []

for i in range(len(ROW)):
    for j in range(len(COL)):
        current_position = str(ROW[i]) + str(COL[j])
        pos_str.append(current_position)

pos_neighbour_dict = dict()

def find_neighbours(var):
    index = pos_str.index(var)
    
    row = index // 9
    col = index % 9
    
    
    neighbours = []
    
    for i in range(9):
        new_row_neighbour = ROW[row] + str(COL[i])
        if new_row_neighbour == var:
            continue
        neighbours.append(new_row_neighbour)


    for i in range(9):
        new_col_neighbour = ROW[i] + str(COL[col])
        if new_col_neighbour == var:
            continue
        neighbours.append(new_col_neighbour)

    box_row_start = row // 3 * 3
    box_col_start = col // 3 * 3
    
    for i in ROW[box_row_start : box_row_start + 3]:
        for j in COL[box_col_start : box_col_start + 3]:
            new_box_neighbour = i + str(j)
            if new_box_neighbour == var:
                continue    
            neighbours.append(new_box_neighbour)
    
    return neighbours

for pos in pos_str:
    pos_neighbour_dict[pos] = find_neighbours(pos)


class sudoku_csp:
    
    # define a class for our csp problems
    
    # define the default constructor
    def __init__(self, board):
        
        # member variables
        self.config = dict()
        
        
        for i in range(len(ROW)):
            for j in range(len(COL)):
                current_position = str(ROW[i]) + str(COL[j])
                
                if board[current_position] == 0:
                    current_domain = COL
                else:
                    current_domain = str(board[current_position])
                self.config[current_position] = current_domain
                
        

    
    
def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)


def backtracking(board):
    """Takes a board and returns solved board."""
    current_csp = sudoku_csp(board)
    assignment = dict()       
    solved_board = backtracking_helper(assignment, current_csp)
    return solved_board


def is_a_valid_solution(assignment):
    return len(assignment) == len(pos_str)

def MRV_heuristic(assignment, csp):
    
    min_pos = None
    min_length = 100
    
    for pos in csp.config:
        if min_length == 1:
            return min_pos
        if pos not in assignment:
            if min_length > len(csp.config[pos]):
                min_length = len(csp.config[pos])
                min_pos = pos
            
    return min_pos




def is_a_valid_board(var, val, assignment): 
    
    for neighbour in pos_neighbour_dict[var]:
        if neighbour in assignment and assignment[neighbour] == val:
            return False
            
    return True

def forwardchecking(var, val, next_csp, assignment):
    
    for neighbour in pos_neighbour_dict[var]:
        if neighbour not in assignment:
            
            # Minimise the domains of all neighbours
            next_csp.config[neighbour] = next_csp.config[neighbour].replace(val, '')
            if len(next_csp.config[neighbour]) == 0:
                return False
            
    return True
    

def backtracking_helper(assignment, csp):
    result = None
    if is_a_valid_solution(assignment):
        return assignment
    
    
    var = MRV_heuristic(assignment, csp)
    
    for val in csp.config[var]:
        
        if is_a_valid_board(var, val, assignment):
            
            next_csp = copy.deepcopy(csp)
            next_csp.config[var] = val
            assignment[var] = val
            

            if forwardchecking(var, val, next_csp, assignment):
                
                result = backtracking_helper(assignment, next_csp)
                
                if result is not None:
                    
                    return result
                
    del assignment[var]
        
    return None


if __name__ == '__main__':
    if len(sys.argv) > 1:
        
        # Running sudoku solver with one board $python3 sudoku.py <input_string>.
        print(sys.argv[1])
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = { ROW[r] + COL[c]: int(sys.argv[1][9*r+c])
                  for r in range(9) for c in range(9)}       
        print_board(board)
        solved_board = backtracking(board)
        
        print_board(solved_board)
        
        # Write board to file
        out_filename = 'output.txt'
        with open(out_filename, 'w') as f:
            f.write(board_to_string(solved_board))
            f.write('\n')
            f.close()

    else:
        # Running sudoku solver for boards in sudokus_start.txt $python3 sudoku.py

        #  Read boards from source.
        src_filename = 'sudokus_start.txt'
        try:
            srcfile = open(src_filename, "r")
            sudoku_list = srcfile.read()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()

        # Setup output file
        out_filename = 'output.txt'
        
        list_of_solving_times = []
        f = open(out_filename, "w")
        # Solve each board using backtracking
        for line in sudoku_list.split("\n"):
            
            if len(line) < 9:
                continue

            # Parse boards to dict representation, scanning board L to R, Up to Down
            board = { ROW[r] + COL[c]: int(line[9*r+c])
                      for r in range(9) for c in range(9)}
            
            # Print starting board. TODO: Comment this out when timing runs.
            print_board(board)
            start_time = time.time()
            # Solve with backtracking
            solved_board = backtracking(board)
            end_time = time.time()
            total_time = end_time - start_time
            list_of_solving_times.append(total_time)
            # Print solved board. TODO: Comment this out when timing runs.
            print_board(solved_board)

            # Write board to file
            
            f.write(board_to_string(solved_board))
            f.write('\n')
        f.close()
        print("Finishing all boards in file.")
 
        list_of_solving_times = np.asarray(list_of_solving_times)        
        file = open("README.txt", "w")
        file.write("Number of boards solved: " + str(list_of_solving_times.shape[0]) + "\n")
        file.write("Min running time: " + str(format(np.min(list_of_solving_times), '.3f')) + "s \n")
        file.write("Max running time: " + str(format(np.max(list_of_solving_times), '.3f')) + "s \n")
        file.write("Mean running time: " + str(format(np.mean(list_of_solving_times), '.3f')) + "s \n")
        file.write("The Standard deviation of running time: " + str(format(np.std(list_of_solving_times), '.3f')) + "s \n")
        file.close()
       