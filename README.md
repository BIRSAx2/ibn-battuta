# Ibn-battuta.rs

This project is a Rust-based implementation of algorithms to solve the Traveling Salesman Problem (TSP). The TSP is a
classic combinatorial optimization problem where the goal is to find the shortest possible tour that visits a set of
cities and returns to the starting city.

## Getting Started

## Project Structure

- data/tsplib: Contains a collection of TSP instances in the TSPLIB format. These instances are used for testing and
  benchmarking the algorithms.
- src/algorithms: Implements various TSP solving algorithms, such as Ant Colony Optimization and Nearest Neighbor.
- src/parser: Provides modules for parsing TSP instances, handling metrics, and managing errors related to TSP
  instances.
- src/utils: Contains utility functions and modules for handling points and other common functionalities.
- tests: Includes test files for the nearest neighbor algorithm.