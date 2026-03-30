# ibn-battuta

`ibn-battuta` is a Rust library for parsing TSPLIB instances and solving the Travelling Salesman Problem with exact, heuristic, and metaheuristic algorithms.

## Status

- Library-first crate
- `cargo test`, doctests, and `cargo clippy --all-targets -- -D warnings` are expected to pass
- Malformed TSPLIB input returns `ParseTspError` instead of panicking in parser entrypoints

## Installation

```toml
[dependencies]
ibn_battuta = "0.1.1"
```

## Quick Start

```rust
use ibn_battuta::{NearestNeighbor, TspBuilder, TspSolver, TwoOpt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tsp = TspBuilder::parse_path("data/tsplib/berlin52.tsp")?;

    let mut nn = NearestNeighbor::new(tsp.clone());
    let base = nn.solve();

    let mut solver = TwoOpt::from(tsp, base.tour, false);
    let solution = solver.solve();

    println!("tour length: {}", solution.length);
    println!("cities: {}", solution.tour.len());

    Ok(())
}
```

## Public API

- Parsing: `TspBuilder`, `Tsp`, `ParseTspError`
- Stable exact solvers: `BellmanHeldKarp`, `BranchAndBound`, `BruteForce`
- Stable heuristics: `NearestNeighbor`, `TwoOpt`
- Additional heuristics and metaheuristics remain available under `ibn_battuta::experimental`
- Distance helpers are exposed under `ibn_battuta::metric`

## Stability Policy

- Crate-root exports are the stable, production-facing surface
- `ibn_battuta::experimental` contains opt-in advanced solvers that remain available but are not the default supported API
- Parser entrypoints are expected to fail with `ParseTspError` on malformed input rather than panic
- Checked read access is available through `Tsp::node` and `Tsp::try_weight`

## Solver Support Matrix

- Stable: `BellmanHeldKarp`, `BranchAndBound`, `BruteForce`, `NearestNeighbor`, `TwoOpt`
- Experimental: `LinKernighan`, `ThreeOpt`, `SimulatedAnnealing`, `SA2Opt`, `GeneticAlgorithm`, `GA2Opt`, `AntSystem`, `AntColonySystem`, `ACS2Opt`, `RedBlackACS`, `RBACS2Opt`
- Benchmark output now includes a `support` column so stable and experimental runs are distinguishable in CSV/stdout output

## TSPLIB Support

- Supported: common coordinate-based and explicit-weight TSPLIB inputs used by the test suite
- Unsupported features return explicit errors instead of placeholder panics
- Explicitly unsupported today: adjacency-list edge parsing and placeholder/path examples that are not real TSPLIB files

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under either MIT or Apache-2.0.
