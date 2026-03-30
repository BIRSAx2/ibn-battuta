//! Experimental solvers and advanced algorithms.
//!
//! These APIs remain available, but they are not part of the crate's stable-by-default surface.
//! Prefer the crate-root exports for production use unless you specifically need one of these
//! algorithms and are comfortable validating it for your workload.

pub use crate::algorithms::heuristic::{LinKernighan, ThreeOpt};
pub use crate::algorithms::metaheuristic::{
    ACS2Opt, AntColonySystem, AntSystem, GA2Opt, GeneticAlgorithm, RBACS2Opt, RedBlackACS, SA2Opt,
    SimulatedAnnealing,
};
