use crate::parser::Tsp;
use std::fmt::{Display, Formatter};
use std::time::Duration;
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
    GeneticAlgorithm2Opt,
    AntSystem,
    AntColonySystem,
    AntColonySystem2Opt,
    RedBlackAntColonySystem,
    RedBlackAntColonySystem2Opt,
    SimulatedAnnealing2Opt,
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
            Solver::AntSystem => write!(f, "AntSystem"),
            Solver::AntColonySystem => write!(f, "AntColonySystem"),
            Solver::RedBlackAntColonySystem => write!(f, "RedBlackAntColonySystem"),
            Solver::RedBlackAntColonySystem2Opt => write!(f, "RedBlackAntColonySystem"),
            Solver::GeneticAlgorithm2Opt => write!(f, "GeneticAlgorithm2Opt"),
            Solver::AntColonySystem2Opt => write!(f, "AntColonySystem2Opt"),
            Solver::SimulatedAnnealing2Opt => write!(f, "SimulatedAnnealing2Opt"),
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