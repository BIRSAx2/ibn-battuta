//! Heuristic solvers.
//!
//! [`NearestNeighbor`] and [`TwoOpt`] are part of the stable root API.
//! [`ThreeOpt`] and [`LinKernighan`] are available through [`crate::experimental`].

pub mod local_search;
pub mod nearest_neighbor;

pub use local_search::*;
pub use nearest_neighbor::*;
