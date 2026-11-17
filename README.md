# Strategic_Wordle_Hosts
Problem. The original Wordle is a popular game where players get 6 chances to guess a 5 letter word. This can also be modeled as a two-player zero-sum game where the host picks a secret word and the guesser tries to find the word in the fewest possible guesses. There are two versions of the host choosing a word that have already been studied. The first one is where the host picks a word uniformly at random, from the 2 315-word answer list, and in this situation, the optimal guesser needs 3.421 guesses on average [1, 2]. The second option for the host is to cheat and switch the word during the ongoing game while staying consistent with the hints the guesser has already uncovered. This version of the game is known as Absurdle [3] and it can always force the guesser to need 5 tries to get the word, and no strategy can solve every word in 4 tries [2]. Combining these two ideas, the proposed study is to analyze a distribution p such that an honest host commits to a probability distribution p over the words before the game begins, but these words are specifically chosen to make an optimal guesser need 5 guesses to solve the puzzle. First we will compute the equilibrium host distribution p* and its value V* , the expected number of guesses given an optimal guesser, and then we’ll ask LLMs to to act as hosts and measure how close they come to V*.

## Code

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/run_double_oracle.py --n 50 --seed 0 --out results/main/n50_s0.json
.venv/bin/python scripts/summarize.py results/main                       # table + summary.csv
```

- `wordle/feedback.py`: Wordle feedback (duplicate letters handled), vectorised pattern matrix `P[guess, answer]`.
- `wordle/solver.py`: `Game` (word subset + guess list) and `best_response(game, p)`, an exact p-weighted DP with branch and bound. Returns the tree and its cost vector `c(T, w)` for every word.
- `wordle/oracle.py`: `double_oracle(game)`. Restricted LP over generated trees (and, by default, a growing host word set) gives host `p` and guesser mix `q`; the best-response tree to `p` gives lower bound L and the hardest word against `q` gives upper bound U, so `L <= V* <= U` at every iteration.
- Defaults: guesser may use all 12,972 allowed words (`--guesses answers` for fast debugging); no 6-guess cap (runs report max tree depth and warn if it exceeds 6).
- `data/`: official answer list (2,315) and allowed guesses (12,972).
