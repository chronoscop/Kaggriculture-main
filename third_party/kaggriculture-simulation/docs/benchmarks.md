# Benchmarks

All numbers were measured on one Windows 11 desktop (24 hardware threads)
that was busy with other work at the same time, using at most 2 worker
threads. They are indicative only; re-measure on your machine:

```
python -m kaggsim.benchmark --games 20 --workers 2      # official vs Rust, same games
kagg bench <two-seat tape> 500                          # raw step function
```

## Same games, official vs Rust

`python -m kaggsim.benchmark --games 20 --workers 2` plays 20 games of the
`scripted` vs `random` fixture policies (Python) on every path and checks
the final banks: **identical for all 20 games on every path**.

| Path | Workers | s / game | games / s | Speed-up vs official x1 |
|---|---:|---:|---:|---:|
| official `env.run` (Python engine, Python agents) | 1 | 2.52 | 0.40 | 1.0 |
| official `env.run`, process pool | 2 | 1.41 | 0.71 | 1.8 |
| `kagg tournament` (Rust engine, same Python agents in hosts) | 1 | 0.30 | 3.3 | 8.3 |
| `kagg tournament` | 2 | 0.15 | 6.6 | 16.5 |

With Python agents the time is dominated by the agents themselves; the
engine and the orchestration are a small fraction.

## Engine-bound paths (no Python)

| Path | Threads | Throughput |
|---|---:|---|
| `kagg bench` (step function, one episode replayed 500 times) | 1 | about 550,000 steps/s, about 770 episodes/s |
| `kagg batch` (2,000 tape-pair jobs, including tape parsing) | 1 | about 450 games/s |
| `kagg batch` | 2 | about 850 games/s |
| `kagg tournament`, built-in fixture policies (200 games) | 2 | 145 to 390 games/s (varied with machine load) |

## Memory

Peak resident set size, sampled every 50 ms with `psutil`:

| Run | Peak RSS |
|---|---|
| `kagg tournament`, 1,000 built-in games, 2 workers | about 7 MiB (the `kagg` process only) |
| `kagg tournament`, 40 games with Python agents, 2 workers | about 8 MiB for `kagg` plus about 22 MiB per Python host (one host per worker), about 51 MiB in total |

Memory grows with the number of workers, not with the number of games:
results stream to disk, and samples are buffered for one game at a time
per worker before they are written.
