



def winning_move(game_state, next_player):
  # given a game_state, find a winning move for next_player
  for move in game_state.legal_moves(next_player):
    next_state = copy(game_state)
    next_state.apply_move(move)
    if next_state.is_over() and next_state.winner == next_player:
      return move
  return None

def eliminate_losing_moves(game_state, next_player):
  non_losing_moves = []
  opponent = next_player.other()
  for move in game_state.legal_moves(next_player):
    next_state = copy(game_state)
    next_state.apply_move(move)
    if not winning_move(next_state, opponent):
      non_losing_moves.append(move)
  return non_losing_moves

def win_in_two_moves(game_state, next_player):
  # find a move that sets up a subsequent winning move that your opponent can't block
  opponent = next_player.other()
  possible_moves = set(game_state.legal_moves(next_player))
  for move in game_state.legal_moves(next_player):
    next_state = copy(game_state)
    next_state.apply_move(move)
    for blocking_move in eliminate_losing_moves(next_state, opponent):
      nnext_state = copy(next_state)
      nnext_state.apply_move(blocking_move)
      winner = winning_move(nnext_state, next_state)
      if winner:
        return winner
    possible_moves.remove(move)
  return possible_moves

# their solution
def find_two_step_win(game_state, next_player):
  opponent = next_player.other()
  for move in game_state.legal_moves(next_player):
    next_state = copy(game_state)
    next_state.apply_move(move)
    # get moves that will block next move, if it's a winner
    good_responses = eliminate_losing_moves(next_state, opponent):
    if not good_responses:
      return move
  return None
