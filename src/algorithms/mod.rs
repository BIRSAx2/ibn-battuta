//! Solver implementations for the travelling salesman problem.
//!
//! Stable-by-default solvers are exposed at the crate root.
//! Additional heuristics and metaheuristics remain available through [`crate::experimental`].

pub mod exact;
pub mod heuristic;
pub mod metaheuristic;
pub mod utils;

pub use exact::*;
pub use heuristic::*;
pub use metaheuristic::*;

/// Represents a solution to the Traveling Salesman Problem (TSP).
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    /// The tour representing the order of nodes visited.
    pub tour: Vec<usize>,
    /// The total length of the tour.
    pub length: f64,
}

impl Default for Solution {
    /// Creates a default `Solution` with an empty tour and zero length.
    fn default() -> Self {
        Solution {
            tour: vec![],
            length: 0.0,
        }
    }
}

impl Solution {
    /// Creates a new `Solution` with the specified tour and total length.
    ///
    /// # Arguments
    ///
    /// * `tour` - A vector of node indices representing the tour.
    /// * `total` - The total length of the tour.
    ///
    /// # Returns
    ///
    /// A new `Solution` instance.
    pub fn new(tour: Vec<usize>, total: f64) -> Self {
        Solution {
            tour,
            length: total,
        }
    }
}

/// A trait for solving the Traveling Salesman Problem (TSP).
pub trait TspSolver {
    /// Solves the TSP and returns a `Solution`.
    ///
    /// # Returns
    ///
    /// A `Solution` struct containing the tour and its total cost.
    fn solve(&mut self) -> Solution;

    /// Returns the tour of the TSP solution.
    ///
    /// # Returns
    ///
    /// A vector of node indices representing the tour.
    fn tour(&self) -> Vec<usize>;

    /// Calculates the cost between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from` - The starting node index.
    /// * `to` - The ending node index.
    ///
    /// # Returns
    ///
    /// The cost between the two nodes.
    fn cost(&self, from: usize, to: usize) -> f64;

    /// Calculates the total cost of a given tour.
    ///
    /// # Arguments
    ///
    /// * `tour` - A vector of node indices representing the tour.
    ///
    /// # Returns
    ///
    /// The total cost of the tour.
    fn calculate_tour_cost(&self, tour: &[usize]) -> f64 {
        let mut total_cost = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total_cost += self.cost(from, to);
        }
        total_cost
    }

    /// Returns the name of the algorithm.
    ///
    /// # Returns
    ///
    /// A string representing the name of the algorithm.
    fn format_name(&self) -> String {
        "TspSolver".to_string()
    }
}

impl std::fmt::Display for dyn TspSolver {
    /// Formats the name of the algorithm for display.
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.format_name())
    }
}

#[cfg(test)]
mod support_tests {
    use super::TspSolver;
    use crate::{
        BellmanHeldKarp, BranchAndBound, BruteForce, NearestNeighbor, Tsp, TspBuilder, TwoOpt,
    };

    fn euclidean_fixture() -> Tsp {
        TspBuilder::parse_str(
            "
NAME : fixture
TYPE : TSP
DIMENSION : 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
  1 0.0 0.0
  2 1.0 0.0
  3 2.0 0.0
  4 2.0 1.0
  5 0.0 1.0
EOF
",
        )
        .unwrap()
    }

    fn explicit_fixture() -> Tsp {
        TspBuilder::parse_str(
            "
NAME : explicit
TYPE : TSP
DIMENSION : 4
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 4 1 9
4 0 6 3
1 6 0 2
9 3 2 0
EOF
",
        )
        .unwrap()
    }

    fn assert_valid_solution(tsp: &Tsp, solver: &dyn TspSolver) {
        let tour = solver.tour();
        assert_eq!(tour.len(), tsp.dim());

        let mut sorted = tour.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..tsp.dim()).collect::<Vec<_>>());

        let reported = solver.calculate_tour_cost(&tour);
        assert!(reported.is_finite());
    }

    #[test]
    fn stable_solvers_return_valid_euclidean_tours() {
        let tsp = euclidean_fixture();

        let mut nearest_neighbor = NearestNeighbor::new(tsp.clone());
        let nn_solution = nearest_neighbor.solve();
        assert_valid_solution(&tsp, &nearest_neighbor);
        assert_eq!(
            nn_solution.length,
            nearest_neighbor.calculate_tour_cost(&nn_solution.tour)
        );

        let mut two_opt = TwoOpt::from(tsp.clone(), nn_solution.tour.clone(), false);
        let two_opt_solution = two_opt.solve();
        assert_valid_solution(&tsp, &two_opt);
        assert_eq!(
            two_opt_solution.length,
            two_opt.calculate_tour_cost(&two_opt_solution.tour)
        );

        let mut held_karp = BellmanHeldKarp::new(tsp.clone());
        let hk_solution = held_karp.solve();
        assert_valid_solution(&tsp, &held_karp);
        assert_eq!(
            hk_solution.length,
            held_karp.calculate_tour_cost(&hk_solution.tour)
        );

        let mut branch_and_bound = BranchAndBound::new(&tsp);
        let bb_solution = branch_and_bound.solve();
        assert_valid_solution(&tsp, &branch_and_bound);
        assert_eq!(
            bb_solution.length,
            branch_and_bound.calculate_tour_cost(&bb_solution.tour)
        );

        let mut brute_force = BruteForce::new(&tsp);
        let brute_solution = brute_force.solve();
        assert_valid_solution(&tsp, &brute_force);
        assert_eq!(
            brute_solution.length,
            brute_force.calculate_tour_cost(&brute_solution.tour)
        );
    }

    #[test]
    fn stable_solvers_handle_explicit_weights() {
        let tsp = explicit_fixture();

        let mut nearest_neighbor = NearestNeighbor::new(tsp.clone());
        let nn_solution = nearest_neighbor.solve();
        assert_valid_solution(&tsp, &nearest_neighbor);
        assert_eq!(
            nn_solution.length,
            nearest_neighbor.calculate_tour_cost(&nn_solution.tour)
        );

        let mut held_karp = BellmanHeldKarp::new(tsp.clone());
        let hk_solution = held_karp.solve();
        assert_valid_solution(&tsp, &held_karp);
        assert_eq!(
            hk_solution.length,
            held_karp.calculate_tour_cost(&hk_solution.tour)
        );

        let mut branch_and_bound = BranchAndBound::new(&tsp);
        let bb_solution = branch_and_bound.solve();
        assert_valid_solution(&tsp, &branch_and_bound);
        assert_eq!(
            bb_solution.length,
            branch_and_bound.calculate_tour_cost(&bb_solution.tour)
        );
    }
}
