use std::fmt::{Display, Formatter};

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
