use std::time::Duration;
use tspf::Tsp;

#[derive(Clone, Debug, PartialEq)]
pub enum Solver {
    BruteForce,
    BranchAndBound,
    NearestNeighbor,
    TwoOpt,
    ThreeOpt,
    Greedy,
    LinKernighan,
    SimulatedAnnealing,
    GeneticAlgorithm,
    AntColonyOptimization,
    AntColonySystem,
    RedBlackAntColonySystem,
}


// Define a struct to hold TSP instance data
pub struct TspInstance {
    pub tsp: Tsp,
    pub best_known: f64,
}

// Define a struct to hold benchmark results
pub struct BenchmarkResult {
    pub instance_name: String,
    pub algorithm_name: String,
    pub execution_time: Duration,
    pub total_cost: f64,
    pub best_known: f64,
    pub solution_quality: f64,
    pub solution: Vec<usize>,
}