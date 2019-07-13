# Usage
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