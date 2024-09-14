# ibn-battuta: A Rust Library for Solving the Travelling Salesman Problem (TSP)

`ibn-battuta` is a powerful, flexible, and extensible Rust library designed to solve the Travelling Salesman Problem (
TSP) using various algorithms, including exact, heuristic, and metaheuristic methods. The library is designed to handle
both small and large instances of TSP efficiently and is optimized for parallel processing, making it suitable for
performance benchmarking across multiple solvers.

## Features

- **Exact Algorithms**:
    - Bellman-Held-Karp
    - Branch-and-Bound
    - Brute Force

- **Heuristic Algorithms**:
    - Nearest Neighbor
    - 2-Opt Local Search
    - 3-Opt Local Search
    - Lin-Kernighan Heuristic

- **Metaheuristic Algorithms**:
    - Ant Colony System (ACS) and Ant System (AS)
    - Genetic Algorithm (GA)
    - Simulated Annealing (SA)
    - Red-Black Ant Colony System (RB-ACS)
    - Hybrid ACS and 2-Opt (ACS-2Opt, RB-ACS-2Opt, etc.)

- **TSP Parser**:
    - Supports parsing `.tsp` files (TSPLIB format) for loading TSP instances.

- **Parallel Processing**:
    - Uses the `rayon` crate for multi-threaded execution, allowing for efficient benchmarking and parallel
      computations.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
ibn-battuta = { git = "https://github.com/your-username/ibn-battuta" }
rayon = "1.5"
```

Then, in your Rust code:

```rust
use ibn_battuta::algorithms::utils::Solver;
use ibn_battuta::parser::TspBuilder;
```

## Usage

### Example: Running Benchmarks on Multiple Solvers

The following example demonstrates how to benchmark multiple TSP solvers in parallel:

```rust
use ibn_battuta::algorithms::utils::Solver;
use ibn_battuta::algorithms::*;
use ibn_battuta::parser::TspBuilder;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::fs::OpenOptions;
use std::time::Duration;

fn main() {
	// List of solvers to benchmark
	let solvers = vec![
		Solver::NearestNeighbor,
		Solver::TwoOpt,
		Solver::SimulatedAnnealing,
		Solver::GeneticAlgorithm,
		Solver::AntColonySystem,
		Solver::RedBlackAntColonySystem,
	];

	// Parameters for each solver
	let params = vec![
		vec![], // NN
		vec![], // 2-Opt
		vec![1000.0, 0.999, 0.0001, 1000.0, 100.0], // SA
		vec![100.0, 5.0, 0.7, 0.01, 500.0], // GA
		vec![0.1, 2.0, 0.1, 0.9, 1000.0, 15.0], // ACS
		vec![0.1, 2.0, 0.1, 0.2, 0.9, 1000.0, 15.0], // RB-ACS
	];

	// Number of threads to use
	let num_threads = 8;

	// TSP instances to benchmark
	let instances = vec![
		TspInstance { path: "data/tsplib/eil51.tsp".to_string(), best_known: 426.0 },
		TspInstance { path: "data/tsplib/berlin52.tsp".to_string(), best_known: 7542.0 },
	];

	// CSV file to save results
	let csv_file = Arc::new(Mutex::new(create_csv_file("Benchmark-Results.csv")));

	run_parallel_benchmarks(&instances, &solvers, &params, num_threads, csv_file);
}

fn create_csv_file(filename: &str) -> std::fs::File {
	OpenOptions::new().write(true).create(true).truncate(true).open(filename).expect("Unable to create file")
}
```

## Supported Algorithms

### Exact Algorithms

These algorithms find the optimal solution but may take a long time for large instances:

1. **Bellman-Held-Karp**: Solves TSP using dynamic programming.
2. **Branch-and-Bound**: Optimizes through branching decisions and bounds.
3. **Brute Force**: Exhaustively searches all possible tours (not recommended for large instances).

### Heuristic Algorithms

These algorithms find good (but not necessarily optimal) solutions quickly:

1. **Nearest Neighbor**: Constructs a solution by iteratively choosing the nearest city.
2. **2-Opt and 3-Opt**: Local search algorithms that iteratively swap edges to reduce the total tour cost.
3. **Lin-Kernighan**: A more advanced local search heuristic based on 2-opt and 3-opt moves.

### Metaheuristic Algorithms

These algorithms use more complex strategies to search the solution space:

1. **Ant Colony System (ACS)**: Uses artificial ants to build solutions based on pheromone trails.
2. **Genetic Algorithm (GA)**: Evolves a population of solutions through crossover and mutation.
3. **Simulated Annealing (SA)**: Gradually cools down a solution, allowing worse moves to escape local optima.
4. **Red-Black Ant Colony System (RB-ACS)**: A variant of ACS with red and black pheromone trails for more robust
   search.
5. **Hybrid Methods**: Combines metaheuristics with 2-Opt (e.g., ACS-2Opt, RB-ACS-2Opt).

## Contributing

Contributions to `ibn-battuta` are welcome! Please submit issues or pull requests via GitHub.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is inspired by the work of various TSP solvers and metaheuristics, and is named after the famous explorer
Ibn Battuta, representing the spirit of exploring optimal solutions in vast solution spaces.

Special thanks to the authors of the following libraries and tools:

- [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/): Used for benchmarking TSP instances.
- [teeline](https://github.com/timgluz/teeline): Used as a reference for implementing exact algorithms.
- [tspf-rs](https://github.com/1crcbl/tspf-rs): Used as base for the parser implementation, with modifications to
  accommodate for needs of this library.

Particular thanks to the authors of the following papers :

- Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization.
- Lin, S., & Kernighan, B. W. (1973). An Effective Heuristic Algorithm for the Traveling-Salesman Problem.
- Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2006). The Traveling Salesman Problem: A Computational
  Study.
- Helsgaun, K. (2000). An Effective Implementation of the Lin-Kernighan Traveling Salesman Heuristic.
- Reinelt, G. (1991). TSPLIB - A Traveling Salesman Problem Library.
- Gambardella, L. M., Dorigo, M., & Blum, C. (1999). Ant Colony System: A Cooperative Learning Approach to the Traveling
  Salesman Problem.
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). The Ant System: Optimization by a Colony of Cooperating Agents.