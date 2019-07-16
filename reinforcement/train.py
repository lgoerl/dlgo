import argparse, h5py
from dlgo.agent.rlagent import load_new_policy_agent, load_policy_agent
from dlgo.reinforcement.experience import load_experience



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_filename', '-e', type=str)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.5)
    parser.add_argument('--clipnorm', '-cn', type=float, default=0.075)
    parser.add_argument('--batch_size', 'bs', type=float, default=512)
    parser.add_argument('--agent_path', '-a', type=str, default='./dlgo/agent/latest.hdf5')
    parser.add_argument('--new_agent', '-n', type=bool, default=False)
    args = parser.parse_args()

    if args.new_agent:
        learning_agent = load_new_policy_agent()
    else:
        learning_agent = agent.load_policy_agent(h5py.File(args.agent_path))

    exp_filename = f'./dlgo/reinforcement/experience/{args.exp_filename.replace('h5', '')}.h5'

    for exp_filename in experience_files:
        exp_buffer = load_experience(h5py.File(args.exp_filename))
        learning_agent.train(
            exp_buffer,
            lr=args.learning_rate,
            clipnorm=args.clipnorm,
            batch_size=args.batch_size,
        )
    with h5py.File(updated_agent_filename, 'w') as agent_out:
        learning_agent.serialize(agent_out)