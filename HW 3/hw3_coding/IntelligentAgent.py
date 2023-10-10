# import random
import sys
import time

import numpy as np

from BaseAI import BaseAI


class IntelligentAgent(BaseAI):

    def getMove(self, grid):

        initial_time = time.process_time()
        optimised_move = self.maximise(initial_time, grid, 0, -sys.maxsize, sys.maxsize)
        return optimised_move[0]

    def terminateTestMax(self, available_moves, depth, initial_time):

        # Make sure we do not exceed the time limit
        if time.process_time() - initial_time >= 0.10:
            return True

        # Make sure this is not the end of the game
        if len(available_moves) == 0:
            return True

        # Make sure we do not exceed depth limit
        if depth >= 4:
            return True

        return False

    def terminateTestMin(self, available_cells, depth, initial_time):

        # Make sure we do not exceed the time limit
        if time.process_time() - initial_time >= 0.10:
            return True

        # Make sure this is not the end of the game
        if len(available_cells) == 0:
            return True

        # Make sure we do not exceed depth limit
        if depth >= 4:
            return True

        return False

    def maximise(self, initial_time, grid, depth, alpha, beta):
        # print(depth)

        available_moves = grid.getAvailableMoves()

        if self.terminateTestMax(available_moves, depth, initial_time):
            return (None, self.heuristic(grid))

        max_move = (None, -sys.maxsize)

        for move_list in available_moves:

            cur_grid = move_list[1]

            min_move = self.minimise(initial_time, cur_grid, depth + 1, alpha, beta)

            max_move = (move_list[0], min_move[1]) if min_move[1] > max_move[1] else max_move

            # alpha-beta pruning
            if beta <= max_move[1]:
                break

            alpha = max_move[1] if alpha < max_move[1] else alpha

        return max_move

    def minimise(self, initial_time, grid, depth, alpha, beta):
        # print(depth)

        available_cells = grid.getAvailableCells()

        if self.terminateTestMin(available_cells, depth, initial_time):
            return (None, self.heuristic(grid))

        min_move = (None, sys.maxsize)

        for cell in available_cells:

            grid_2 = grid.clone()
            grid_4 = grid.clone()

            grid_2.insertTile(cell, 2)
            grid_4.insertTile(cell, 4)

            move_2 = self.maximise(initial_time, grid_2, depth + 1, alpha, beta)
            move_4 = self.maximise(initial_time, grid_4, depth + 1, alpha, beta)

            weighted_heuristic = move_2[1] * 0.9 + move_4[1] * 0.1

            min_move = (None, weighted_heuristic) if min_move[1] > weighted_heuristic else min_move

            # alpha-beta pruning
            if alpha >= min_move[1]:
                break

            beta = min_move[1] if beta > min_move[1] else beta

        return min_move

    def heuristic_available_cell_count(self, grid):
        """
        The number of empty cells is the most essential heuristic for the 2048 game
        so, rather than assigning a weight to it, we use it an indepent term, where
        we actually use h_acc * (w_2 * h_2 + w_3 * h3 + ...)
        """

        return len(grid.getAvailableCells())

    def heuristic_available_move_count(self, grid):
        """
        The number of available moves is also the most essential heuristic for the game
        so, rather than assigning a weight to it, we use it an indepent term, where
        we actually use h_amc * (w_2 * h_2 + w_3 * h3 + ...)
        """

        return len(grid.getAvailableMoves())

    def heuristic_mono_smooth_value(self, grid):
        """
        Many 2048 masters puts the largest tile at the corner, they also they to avoid
        large differences between neighbours, the full implementations of Smoothness and
        Monotonicity are to complicated. To minimise runtime, we propose an easy way to
        calculate the two heuristics in a single nested for loop.
        """

        n = 4

        """
        Four possible arrangements of putting the max value at the corner. All these settings
        also highly awards mono-arrangements
        """

        br = np.asarray([[1, 4, 16, 64],
                         [4, 16, 64, 256],
                         [16, 64, 256, 1024],
                         [64, 256, 1024, 4096]])
        #
        # tr = np.asarray([[64, 256, 1024, 4096],
        #                  [16, 64, 256, 1024],
        #                  [4, 16, 64, 256],
        #                  [1, 4, 16, 64]])
        #
        # tl = np.asarray([[4096, 1024, 256, 64],
        #                  [1024, 256, 64, 16],
        #                  [256, 64, 16, 4],
        #                  [64, 16, 4, 1]])
        #
        # bl = np.asarray([[64, 16, 4, 1],
        #                  [256, 64, 16, 4],
        #                  [1024, 256, 64, 16],
        #                  [4096, 1024, 256, 64]])

        # arrangement_list = [br, tr, tl, bl]

        """
        After fine-tuning, we should only consider the bottom-right arrangement.
        By reviewing many 2048 masters' videos, they tend to only move the max
        tile to only one corner. That is, they tend to only move in three directions.
        In this way, they can easily guarantee the monotonicity. So we keep only 
        the bottom-right arrangement to make it easier to calculate and maintain the
        monotonicity of the grid.
        """
        arrangement_list = [br]

        mono_array = np.zeros(len(arrangement_list))

        """
        To align with our bottom right arrangement, our difference, which calculate
        the smoothness of the grid, only consider the bottom cell or the right cell 
        as its neighbours.
        """
        difference = 0

        for i in range(n):
            for j in range(n):
                cur_cell_val = grid.map[i][j]

                """
                Calculate the smoothness 
                """
                diff_i, diff_j = 0, 0
                if i < n - 1:
                    larger, smaller = max(cur_cell_val, grid.map[i + 1][j]), min(cur_cell_val, grid.map[i + 1][j])
                    diff_i = larger - smaller

                if j < n - 1:
                    larger, smaller = max(cur_cell_val, grid.map[i][j + 1]), min(cur_cell_val, grid.map[i][j + 1])
                    diff_j = larger - smaller

                difference = difference + min(diff_i, diff_j)

                """
                Calculate the monotonicity
                """
                for index, item in enumerate(arrangement_list):
                    mono_array[index] = mono_array[index] + cur_cell_val * item[i][j]

        # print(mono_array)
        best_score = np.max(mono_array)
        # print(best_score)
        difference = difference * grid.getMaxTile()
        return best_score, difference

    def heuristic(self, grid):

        h_1 = self.heuristic_available_cell_count(grid)
        h_2 = self.heuristic_available_move_count(grid)
        h_3, h_4 = self.heuristic_mono_smooth_value(grid)
        # print("part 1" +str(h_1 * h_2 * h_3))
        # print(h_4 * grid.getMaxTile() * 32)


        """
        h_1 and h_2 are the most essential part of an 2048 game, the parameter 128.0
        is chosen to rescale h_4, also, we used grid.getMaxTile() to dynamically adjust
        the weight
        """
        # h = h_1 * (h_2 * h_2 * h_3 * 8 - h_4 * grid.getMaxTile() * 128) # never reach 2048
        # 41.3% to reach 2048, 28% to reach 1024, 7% to reach 4096
        # h = h_1 * h_2 * h_3 - h_4 * 96

        h = (h_1 + 1) * h_2 * h_2 * h_3 - h_4 * 512

        return h
