# ibn-battuta

[![crates.io](https://img.shields.io/crates/v/ibn_battuta.svg)](https://crates.io/crates/ibn_battuta)
[![docs.rs](https://docs.rs/ibn_battuta/badge.svg)](https://docs.rs/ibn_battuta)
[![license](https://img.shields.io/crates/l/ibn_battuta.svg)](LICENSE)

`ibn-battuta` is a Rust library for two related jobs: parsing TSPLIB instances and solving the travelling salesman problem with exact, heuristic, and metaheuristic algorithms.

The crate root is the stable library surface. It includes the TSPLIB parser, the `Tsp` model, exact solvers, `NearestNeighbor`, and `TwoOpt`. Additional solvers remain available under `ibn_battuta::experimental`.

## Installation

Add the crate to your project:

```toml
[dependencies]
ibn_battuta = "0.2.0"
```

## Quick start

This example parses a small TSPLIB instance from a string, builds a nearest-neighbor tour, and improves it with 2-opt.

```rust
use ibn_battuta::{NearestNeighbor, TspBuilder, TspSolver, TwoOpt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = "
    NAME : square
    TYPE : TSP
    DIMENSION : 4
    EDGE_WEIGHT_TYPE : EUC_2D
    NODE_COORD_SECTION
    1 0 0
    2 0 1
    3 1 1
    4 1 0
    EOF
    ";

    let tsp = TspBuilder::parse_str(data)?;

    let mut nn = NearestNeighbor::new(tsp.clone());
    let base = nn.solve();

    let mut solver = TwoOpt::from(tsp, base.tour, false);
    let solution = solver.solve();

    println!("tour length: {}", solution.length);
    println!("tour: {:?}", solution.tour);

    Ok(())
}
```

If you already have a `.tsp` file on disk, use `TspBuilder::parse_path("path/to/file.tsp")`.

## Stable API

These types are re-exported at the crate root and are the supported production-facing API:

| Area | Types |
| --- | --- |
| Parsing | `TspBuilder`, `Tsp`, `ParseTspError` |
| Exact solvers | `BellmanHeldKarp`, `BranchAndBound`, `BruteForce` |
| Heuristics | `NearestNeighbor`, `TwoOpt` |
| Common traits and output | `TspSolver`, `Solution` |
| Distance helpers | `metric` |

`Tsp::node` and `Tsp::try_weight` are the checked accessors. They are useful when you want to validate node and edge lookups instead of relying on panicking indexing paths.

## Experimental solvers

Experimental solvers live under `ibn_battuta::experimental`. That module currently includes:

`LinKernighan`, `ThreeOpt`, `SimulatedAnnealing`, `SA2Opt`, `GeneticAlgorithm`, `GA2Opt`, `AntSystem`, `AntColonySystem`, `ACS2Opt`, `RedBlackACS`, and `RBACS2Opt`.

The ant-colony solvers are available like this:

```rust
use ibn_battuta::experimental::{ACS2Opt, RedBlackACS};
use ibn_battuta::{TspBuilder, TspSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tsp = TspBuilder::parse_path("data/tsplib/berlin52.tsp")?;

    let mut acs = ACS2Opt::with_options(tsp.clone(), 0.1, 2.0, 0.1, 0.95, 10, 1000, 20);
    let acs_solution = acs.solve();

    let mut rbacs = RedBlackACS::new(tsp, 0.1, 2.0, 0.1, 0.2, 0.95, 10, 1000, 20);
    let rbacs_solution = rbacs.solve();

    println!("ACS2Opt: {}", acs_solution.length);
    println!("RBACS: {}", rbacs_solution.length);

    Ok(())
}
```

## TSPLIB support

The parser is built around the TSPLIB formats covered by the test suite and benchmarks. Common coordinate-based problems and common explicit-weight inputs are supported. Malformed input returns `ParseTspError` from the parser entry points instead of panicking.

Unsupported or incomplete TSPLIB features return explicit errors. The library does not claim full coverage of every TSPLIB edge format.

## What to use

If you need an exact answer on a small instance, start with `BellmanHeldKarp`, `BranchAndBound`, or `BruteForce`.

If you need a stable approximate solver, start with `NearestNeighbor` and then improve the tour with `TwoOpt`.

If you want stronger heuristic results and are comfortable using the experimental module, `ACS2Opt` and `RBACS2Opt` are the strongest ant-colony hybrids in the current benchmark setup.

## Benchmarks

The repository includes a custom benchmark harness in `benches/benchmarks.rs`. It supports focused, small, and full profiles, CSV output, seeded runs, and solver filtering through environment variables.

Examples:

```bash
cargo bench --bench benchmarks
```

```bash
IBN_BATTUTA_BENCH_PROFILE=focused \
IBN_BATTUTA_BENCH_SOLVERS=ACS,ACS2Opt,RBACS,RBACS2Opt \
cargo bench --bench benchmarks
```

The benchmark output includes solver name, support level, seed, runtime, tour length, best-known optimum, and percentage gap.

## Development

The current quality bar for the library is:

```bash
cargo test
cargo test --doc
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under MIT.
