# Usage
**NOTE** all commands ran in the parent directory of `dlgo`

## Training
Amend the `main_network.py` file as desired.
Training checkpoints are created at each epoch and stored in `checkpoints/`.
##### Editable Hyperparams/things
- `num_games`
- `epochs`
- `batch_size`
- `model`
- `optimizer`
- `encoder`

### Run
```
python -m dlgo.nn.main_network
```

## Run locally
```
python -m dlgo.gtp.play_local
```

## Run reinforcement agent self-play
```
python -m dlgo.reinforcement.simulate_game
```

**options**
- `--num_games -n`: number of games to play; default = 1
- `--in_file -in`: `hdf5` file containing a Keras 'model' group; default = `./dlgo/agent/latest.hdf5`