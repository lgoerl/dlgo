from dlgo import goboard_fast as goboard
from dlgo.agent.base import Agent
from dlgo import scoring

class TerminationStrategy():
    def __init__(self):
        pass
    def should_pass(self, game_state):
        return False
    def should_resign(self, game_state):
        return False

class PassWhenOpponentPasses(TerminationStrategy):
    def should_pass(self, game_state):
        if game_state.last_move is not None:
            return game_state.last_move.is_pass
        else:
            return False
    def should_resign(self, game_state):
        return False

def get(termination):
    if termination == 'opponent_passes':
        return PassWhenOpponentPasses()
    else:
        raise ValueError('Unsupported termination strategy: %s' % termination)

class TerminationAgent(Agent):
    def __init__(self, agent, strategy):
        super().__init__()
        self.agent = agent
        self.strategy = strategy if strategy is not None else TerminationStrategy()

    def select_move(self, game_state):
        if self.strategy.should_pass(game_state):
            return goboard.Move(is_pass=True)
        elif self.strategy.should_resign(game_state):
            return goboard.Move(is_resign=True)
        else:
            return self.agent.select_move(game_state)