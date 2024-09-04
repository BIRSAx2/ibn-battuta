use std::fmt::{Display, Formatter};
use std::time::Duration;
use crate::parser::Tsp;
#[derive(Clone, Debug, PartialEq, Copy)]
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

impl Display for Solver {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Solver::BruteForce => write!(f, "BruteForce"),
            Solver::BranchAndBound => write!(f, "BranchAndBound"),
            Solver::NearestNeighbor => write!(f, "NearestNeighbor"),
            Solver::TwoOpt => write!(f, "TwoOpt"),
            Solver::ThreeOpt => write!(f, "ThreeOpt"),
            Solver::Greedy => write!(f, "Greedy"),
            Solver::LinKernighan => write!(f, "LinKernighan"),
            Solver::SimulatedAnnealing => write!(f, "SimulatedAnnealing"),
            Solver::GeneticAlgorithm => write!(f, "GeneticAlgorithm"),
            Solver::AntColonyOptimization => write!(f, "AntColonyOptimization"),
            Solver::AntColonySystem => write!(f, "AntColonySystem"),
            Solver::RedBlackAntColonySystem => write!(f, "RedBlackAntColonySystem"),
        }
    }
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