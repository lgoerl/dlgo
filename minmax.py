import enum, random
from dlgo.agent import Agent

class MinimaxAgent(Agent):
  def select_move(self, game_state):
    winning_moves = []
    draw_moves = []
    losing_moves = []
    for move in game_state.legal_moves():
      next_state = copy(game_state)
      next_state = next_state.apply_move(move)
      oppo_best_outcome = best_result(next_state)
      agent_best_outcome = reverse_game_result(oppo_best_outcome)
      if agent_best_outcome == GameResult.win:
        winning_moves.append(move)
      elif agent_best_outcome == GameResult.draw:
        draw_moves.append(move)
      else:
        losing_moves.append(move)
    if winning_moves:
      return random.choice(winning_moves)
    elif draw_moves:
      return random.choice(draw_moves)
    return random.choice(losing_moves)

class GameResult(enum.Enum)
  loss = 1
  draw = 2
  win = 3

def reverse_game_result(result):
  # return ((result - 2) * -1) + 2
  return {1:3, 2:2, 3:1}[result]

def best_result0(game_state):
  if game_state.is_over():
    if game_state.winner() == game_state.next_player:
      return GameResult.win
    elif game_state.winner() is None:
      return GameResult.draw
    else:
      return GameResult.loss
  x = GameResult.loss
  for move in game_state.legal_moves():
    next_state = copy(game_state)
    next_state = next_state.apply_move(move)
    oppo_best_outcome = best_result0(next_state)
    player_best_outcome = reverse_game_result(oppo_best_outcome)
    if player_best_outcome.value > x.value:
      x = player_best_outcome.value
  return x

def best_result(game_state, max_depth, eval_fn):
  if game_state.is_over():
    if game_state.winner() == game_state.next_player:
      return MAX_SCORE
    else:
      return MIN_SCORE
  if max_depth == 0:
    return eval_fn(game_state)
  best_so_far = MIN_SCORE
  for move in game_state.legal_moves():
    next_state = copy(game_state)
    next_state = next_state.apply_move(move)
    oppo_best_outcome = best_result(next_state, max_depth, eval_fn)
    player_best_outcome = reverse_game_result(oppo_best_outcome)
    if player_best_outcome > best_so_far:
      best_so_far = player_best_outcome
  return best_so_far

def capture_diff(game_state):
  black_stones = 0
  white_stones = 0
  for r in range(1, game_state.board.num_rows + 1):
    for c in range(1, game_state.board.num_cols + 1):
      p = go_types.Point(r,c)
      color = game_state.board.get(p)
      if color == go_types.Player.black:
        black_stones += 1
      elif color == go_types.Player.white:
        white_stones += 1
  diff = black_stones - white_stones
  if game_state.next_player == go_types.Player.black:
    return diff
  return -1 * diff

def alpha_beta_result(game_state, max_depth, best_black, best_white, eval_fn):
  if game_state.is_over():
    if game_state.winner() == game_state.next_player:
      return MAX_SCORE
    else:
      return MIN_SCORE
  if max_depth == 0:
    return eval_fn(game_state)
  best_so_far = MIN_SCORE
  for move in game_state.legal_moves():
    next_state = copy(game_state)
    next_state = next_state.apply_move(move)
  oppo_best_outcome = alpha_beta_result(
    next_state,
    max_depth-1,
    best_black,
    best_white,
    eval_fn,
  )
  player_best_outcome = -1 * oppo_best_outcome
  if player_best_outcome > best_so_far:
    best_so_far = player_best_outcome
  if game_state.next_player == Player.white:
    if best_so_far > best_white:
      best_white = best_so_far
    outcome_for_black = -1 * best_so_far
    if outcome_for_black < best_black:
      return best_so_far
  if game_state.next_player == Player.black:
    if best_so_far > best_black:
      best_black = best_so_far
    outcome_for_white = -1 * best_so_far
    if outcome_for_white < best_white:
      return best_so_far
  return best_so_far
