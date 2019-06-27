
Important game-specific logic is found in 
- `go_types.Player`
- `goboard.Move`
- `goboard.GameState`: `apply_move`, `legal_moves`, `is_over`, and `winner`

Exercises:
utilizing a method `game_state.legal_moves()`,
1) write a `find_winning_move` method which takes `game_state` and `next_player` as arguments
2) write a `eliminate_losing_moves` method which takes the same. 
3) write a method that will look for a move that sets up a subsequent winning move that your opponent can't block using the above two
