import argparse, h5py, os
from datetime import datetime
from dlgo.agent.rlagent import load_new_policy_agent, load_policy_agent
from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.reinforcement import experience as ex
from dlgo.scoring import compute_game_result

BOARD_SIZE = (19, 19)

def simulate_game(player_b, player_w):
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: player_b,
        Player.white: player_w,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    game_result = compute_game_result(game)
    return game_result

def experience_simulation(num_games, agent_b, agent_w):
    collector_b = ex.ExperienceCollector()
    collector_w = ex.ExperienceCollector()
    for i in range(args.num_games):
        collector_b.begin_episode()
        collector_w.begin_episode()
        agent_b.set_collector(collector_b)
        agent_w.set_collector(collector_w)

        game_record = simulate_game(agent_b, agent_w)

        if game_record.winner == Player.black:
            collector_b.complete_episode(reward=1)
            collector_w.complete_episode(reward=-1)
        else:
            collector_b.complete_episode(reward=-1)
            collector_w.complete_episode(reward=1)

    return ex.combine_experience([
        collector_b, collector_w
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', '-n', type=int, default=1)
    parser.add_argument('--in_file', '-in', type=str, default='./dlgo/agent/latest.hdf5')
    args = parser.parse_args()
    if args.in_file == 'new':
        agent_b = load_new_policy_agent()
        agent_w = load_new_policy_agent()
    else:
        agent_b = load_policy_agent(h5py.File(in_file))
        agent_w = load_policy_agent(h5py.File(in_file))
    dstring = datetime.now().strftime('%Y%m%d%H%M')
    out_path = f'./dlgo/reinforcement/experience/{args.num_games}played{dstring}.h5'
    out_file = args.out_file if args.out_file is not None else out_path

    experience = experience_simulation(args.num_games, agent_b, agent_w)
    with h5py.File(out_file, 'w') as out:
        experience.serialize(out)