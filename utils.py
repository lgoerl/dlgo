import numpy as np
from dlgo.gotypes import Player, Point

COLS = 'ABCDEFGHIJKLMNOPQRST'
STONE_TO_CHAR = {
  None: ' . ',
  Player.black: ' x ',
  Player.white: ' o ',
}

def print_move(player, move):
  if move.is_pass:
    move_str = 'passes'
  elif move.is_resign:
    move_str = 'resigns'
  else:
    move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
  print(move_str)

def print_board(board):
  for row in range(board.num_rows, 0, -1):
    bump = ' ' if row <= 9 else ''
    line = []
    for col in range(1, board.num_cols + 1):
      stone = board.get(Point(row=row, col=col))
      line.append(STONE_TO_CHAR[stone])
    print('%s%d %s' % (bump, row, ''.join(line)))
  print('    ' + '  '.join(COLS[:board.num_cols]))
  print('    ' + ''.join([str(i)+'   '[:-len(str(i))] for i in range(1,board.num_cols+1)]))

def point_from_coords(coords):
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return Point(row=row, col=col)

def coords_from_point(point):
    return '%s%d' % (COLS[point.col-1], point.row)

class MoveAge():
    def __init__(self, board):
        self.move_ages = -np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row-1, point.col-1] = -1

    def add(self, point):
        self.move_ages[point.row-1, point.col-1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1

# from dlgo import goboard_fast as g
# from dlgo import utils
# from dlgo.gotypes import Point
# board = g.Board(19,19)
# state = g.GameState.new_game(19)
# for p in [(2,3),(2,10),(3,2),(3,12),(2,1),(2,1),(1,1),(1,3),(1,4)]:
#     state = state.apply_move(g.Move.play(Point(p)))
# utils.print_board(state.board)

# state = g.GameState.new_game(19)
# for p in [(2,3),(2,4),(6,2),(2,2),(6,1),(3,3),(6,3)]:
#     state = state.apply_move(g.Move.play(Point(*p)))

# utils.print_board(state.board)
# state.board._grid[Point(2,3)].liberties
# state = state.apply_move(g.Move.play(Point(1,3)))