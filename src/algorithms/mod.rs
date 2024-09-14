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
	fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
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
		format!("{}", "TspSolver")
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