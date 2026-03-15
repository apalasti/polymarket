# autoresearch - Polymarket Strategy Optimization

This is an experiment to have the LLM autonomously research and optimize a Polymarket crypto trading strategy.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The codebase is small. Read these files for full context:
   - `findings/project-overview.md` — overall project architecture and data flow.
   - `scripts/crypto_backtest.py` — the backtesting engine (fixed). Evaluates the strategy against historical orderbooks.
   - `src/strategy.py`, `src/execution.py`, `src/transaction.py` — core classes and execution logic (fixed).
   - `src/ai_strategy.py` — the file you modify. Contains the `AIStrategy` class that decides when to buy and sell.
4. **Verify data exists**: Check that the required DuckDB databases (`data/crypto-prices/train.duckdb` for training models and `data/crypto-prices/test.duckdb` for backtesting) exist and the framework is ready to run.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a full backtest across historical Polymarket crypto markets (15-minute up/down markets). You launch it simply as: `uv run python -m scripts.crypto_backtest`.
Always use `uv` for running scripts.

**What you CAN do:**
- **Modify `src/ai_strategy.py`** — this is the main file you edit. Everything inside `AIStrategy` is fair game: modifying thresholds, position sizing, utilizing the orderbook depth (`MarketState`), time remaining (`secs_from_start`), underlying asset price (`price`), and parsing any external probability states.
- **Train Machine Learning Models**: You have access to `data/crypto-prices/train.duckdb`. You can create new Python scripts (e.g. `scripts/train_model.py`) that query this database to extract features, train ML models (like `scikit-learn`, `xgboost`, `lightgbm`, or basic linear models), and save them (e.g. as `.pkl` or `.joblib` files) **very important output these into the `models/` directory**.
- **Load Models**: Modify `src/ai_strategy.py` to load your trained model during `__init__` and use its predictions within your `process_orderbook` logic.

**What you CANNOT do:**
- Modify `scripts/crypto_backtest.py`. It is read-only. It contains the fixed backtest engine, portfolio valuation, and metrics calculation.
- Modify the `src/strategy.py`, `src/execution.py`, or `src/transaction.py` files. These strictly define the simulation rules.
- Install new packages or add dependencies without user approval.

**The goal is simple: maximize `Sharpe ratio` while maintaining a positive `Total return` and keeping `Max drawdown` below 20%.** A 15% return with a 0.5 Sharpe is worse than an 8% return with a 2.5 Sharpe. You are trying to find the most profitable and consistent trading rules that generalize well. Everything is fair game: when to enter, when to exit early, sizing trades based on confidence, or ignoring low-liquidity books.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity or curve-fits to specific market slugs is not worth it. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. Overfitting to historical noise will destroy real-world performance. Use `train.duckdb` exclusively for training ML models, and `test.duckdb` (which `scripts/crypto_backtest.py` uses) exclusively for validation.

**CRITICAL WARNINGS FOR ML TRAINING:**
1. **Prevent Lookahead Bias:** When extracting features from `train.duckdb`, you MUST NOT use future data. Only use orderbook and price data available exactly at or before the timestamp you are predicting. Do not join `outcome_price` into your features!
2. **Model Artifacts:** Do NOT commit `.pkl` or `.joblib` files to Git (this is already handled via `.gitignore`). Ensure your `train_model.py` script overwrites the model file every time it runs so your `ai_strategy.py` doesn't load a stale model after a `git reset`.
3. **Speed:** If `train.duckdb` is massive, downsample it (e.g., use a fraction of slugs) when training models to ensure your iteration loop stays under 5 minutes.
4. **Optimization Metric:** Prioritize **Sharpe Ratio** over Total Return. Discard strategies that take catastrophic risks (Max Drawdown > 20%).

**The first run**: Your very first run should always be to establish the baseline, so you will run the `scripts/crypto_backtest.py` script as is.

## Output format

Once the script finishes it prints a summary like this:

```
--- Backtest summary ---
Markets: 150
Total return: 12.50%
Mean period return: 0.0833%
Volatility (std): 1.5000%
Sharpe ratio: 1.85
Win rate: 55.0%
Max drawdown: 4.50%
```

You can extract the key metrics from the log file:

```bash
grep "^Total return:\|^Sharpe ratio:\|^Win rate:\|^Max drawdown:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	total_return	sharpe_ratio	status	description
```

1. git commit hash (short, 7 chars)
2. total_return achieved (e.g. 12.50 — drop the % sign) — use 0.00 for crashes
3. sharpe_ratio achieved (e.g. 1.85) — use 0.00 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	total_return	sharpe_ratio	status	description
a1b2c3d	12.50	1.85	keep	baseline
b2c3d4e	15.20	2.10	keep	only trade when edge > 5%
c3d4e5f	11.00	1.40	discard	scale position size by time to expiry
d4e5f6g	0.00	0.00	crash	syntax error in strategy logic
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `src/ai_strategy.py` or write/update training scripts (`scripts/train_model.py`) with an experimental idea. Think about market mechanics: 15-minute markets converge to 0 or 1 at expiry. Edge usually comes from discrepancies between orderbook implied probabilities and actual statistical probabilities.
3. git commit
4. Run the experiment: 
   - If your strategy relies on an ML model that needs retraining, run it first: `python scripts/train_model.py` (or using `uv run`)
   - Run the backtest: `uv run python -m scripts.crypto_backtest > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^Total return:\|^Sharpe ratio:\|^Max drawdown:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If `sharpe_ratio` improved (higher is better) AND `Max drawdown` is below 20%, you "advance" the branch, keeping the git commit.
9. If `sharpe_ratio` is equal or worse (or if drawdown exceeds 20%), you `git reset --hard HEAD~1` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each backtest should complete reasonably fast. If a run stalls, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (exception in strategy execution, bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken or violates execution rules (like spending more capital than available), just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the state dicts, utilize `secs_from_start`, read both the bid and ask sides of the book to calculate spread and midpoint, try more complex entry/exit logic, or train new models on `train.duckdb`. The loop runs until the human interrupts you, period.
