use std::fmt::{Display, Formatter};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverSupport {
    Stable,
    Experimental,
}

impl Display for SolverSupport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "stable"),
            Self::Experimental => write!(f, "experimental"),
        }
    }
}

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

impl Solver {
    pub fn support(self) -> SolverSupport {
        match self {
            Self::BruteForce | Self::BranchAndBound | Self::NearestNeighbor | Self::TwoOpt => {
                SolverSupport::Stable
            }
            Self::ThreeOpt
            | Self::Greedy
            | Self::LinKernighan
            | Self::SimulatedAnnealing
            | Self::GeneticAlgorithm
            | Self::GeneticAlgorithm2Opt
            | Self::AntSystem
            | Self::AntColonySystem
            | Self::AntColonySystem2Opt
            | Self::RedBlackAntColonySystem
            | Self::RedBlackAntColonySystem2Opt
            | Self::SimulatedAnnealing2Opt => SolverSupport::Experimental,
        }
    }
}

pub fn should_parallelize(work_items: usize) -> bool {
    work_items > 1 && available_parallelism() > 1
}

pub fn available_parallelism() -> usize {
    static PARALLELISM: OnceLock<usize> = OnceLock::new();
    *PARALLELISM.get_or_init(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    })
}
