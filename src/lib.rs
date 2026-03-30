//! `ibn-battuta` is a library for parsing TSPLIB instances and solving TSP variants.
//!
//! Stable, library-first entrypoints are re-exported at the crate root:
//! [`TspBuilder`], [`Tsp`], [`ParseTspError`], [`BellmanHeldKarp`], [`BranchAndBound`],
//! [`BruteForce`], [`NearestNeighbor`], and [`TwoOpt`].
//!
//! Additional heuristics and metaheuristics remain available under [`experimental`].
//!
//! # Stability
//!
//! The crate root is the supported production surface.
//! Algorithms in [`experimental`] remain available, but they should be treated as opt-in advanced
//! solvers rather than stable-by-default building blocks.
//!
//! # Checked Access
//!
//! [`Tsp::node`] and [`Tsp::try_weight`] provide non-panicking access for callers that need to
//! validate inputs before using solver internals.
#![forbid(unsafe_code)]

pub mod algorithms;
pub mod experimental;
pub mod parser;

pub use algorithms::{
    exact::{BellmanHeldKarp, BranchAndBound, BruteForce},
    heuristic::{NearestNeighbor, TwoOpt},
    Solution, TspSolver,
};
pub use parser::{
    metric, CoordKind, DisplayKind, EdgeFormat, ParseTspError, Point, Tsp, TspBuilder, TspKind,
    WeightFormat, WeightKind,
};
