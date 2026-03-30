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
- Additional heuristics and metaheuristics remain available through the crate API
- Distance helpers are exposed under `ibn_battuta::metric`

## TSPLIB Support

- Supported: common coordinate-based and explicit-weight TSPLIB inputs used by the test suite
- Unsupported features return explicit errors instead of placeholder panics
- `Tsp::try_weight` and `Tsp::node` provide non-panicking read access for callers that need checked access

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under either MIT or Apache-2.0.
