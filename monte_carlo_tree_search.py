import random
from agent.base import Agent

class MCTSNode(object):
  """docstring for MCTSNode"""
  def __init__(self, game_state, parent=None, move=None):
    super(MCTSNode, self).__init__()
    self.game_state = game_state
    self.parent = parent
    self.move = move
    self.win_counts = {
      Player.black: 0,
      Player.white: 0,
    }
    self.num_rollouts = 0
    self.children = []
    self.unvisited_moves = game_state.legal_moves()

  def add_random_child(self):
    index = random.randint(0, len(self.unvisited_moves)-1)
    new_move = self.unvisited_moves.pop(index)
    new_game_state = self.game_state.apply_move(new_move)
    new_node = MCTSNode(new_game_state, parent=self, move=new_move)
    self.children.append(new_node)
    return new_node

  def record_win(self, winner):
    self.win_counts[winner] += 1
    self.num_rollouts += 1

  def can_add_child(self):
    return len(self.unvisited_moves) > 0

  def is_terminal(self):
    return self.game_state.is_over()

  def winning_pct(self, player):
    return float(self.num_rollouts)/self.win_counts[player]

class MCTSAgent(agent.Agent):
  def select_move(self, game_state, rounds=10):
    root = MCTSNode(game_state)

    for i in range(rounds):
      node = root
      while (not node.can_add_child()) and (not node.is_terminal()):
        node = self.select_child(node)

      if node.can_add_child():
        node = node.add_random_child()

      winner = self.simulate_random_game(node.game_state)

      while node is not None:
        node.record_win(winner)
        node = node.parent

    best_move = None
    best_pct = -1.0

    for child in root.children:
      child_pct = child.winning_pct(game_state.next_player)
      if child_pct > best_pct:
        best_pct = child_pct
        best_move = child.move
    return best_move

  def select_child(self, node):
    total_rollouts = sum(child.num_rollouts for child in node.children)
    best_score = -1
    best_child = None
    for child in node.children:
      score = uct_score(
        total_rollouts,
        child.num_rollouts,
        child.winning_pct(node.game_state.next_player),
        self.temperature,
      )
      if score > best_score:
        best_score = score
        best_child = child
    return best_child


def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
  exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
  return win_pct + temperature * exploration
