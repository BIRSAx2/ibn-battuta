use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;

#[allow(dead_code)]
/// A Lin-Kernighan heuristic solver for the Travelling Salesman Problem (TSP).
/// This solver tries to improve an initial tour by performing local optimizations.
///
/// # Fields
/// * `tsp` - The TSP instance containing the distances between cities.
/// * `tour` - The current tour (order of visiting the cities).
/// * `cost` - The current cost of the tour.
/// * `verbose` - If true, print additional information during the solving process.
/// * `max_iterations` - Maximum number of iterations to attempt tour improvement.
pub struct LinKernighan {
	tsp: Tsp,
	tour: Vec<usize>,
	cost: f64,
	verbose: bool,
	max_iterations: usize,
}

impl LinKernighan {
	/// Creates a new `LinKernighan` solver for the given `tsp`.
	/// Initializes the tour randomly.
	///
	/// # Arguments
	/// * `tsp` - The TSP instance.
	///
	/// # Returns
	/// A new instance of the Lin-Kernighan solver.
	pub fn new(tsp: Tsp) -> LinKernighan {
		let mut result = LinKernighan {
			tsp,
			tour: vec![],
			cost: 0.0,
			verbose: false,
			max_iterations: 1000,
		};

		result.initial_tour();

		result
	}

	/// Creates a `LinKernighan` solver with additional options.
	///
	/// # Arguments
	/// * `tsp` - The TSP instance.
	/// * `base_tour` - The initial tour to start the search from.
	/// * `verbose` - Enables or disables verbose output.
	/// * `max_iterations` - Sets the maximum number of iterations to attempt.
	///
	/// # Returns
	/// A new instance of the Lin-Kernighan solver.
	pub fn with_options(tsp: Tsp, base_tour: Vec<usize>, verbose: bool, max_iterations: usize) -> LinKernighan {
		LinKernighan {
			tsp,
			tour: base_tour.clone(),
			cost: 0.0,
			verbose,
			max_iterations,
		}
	}

	/// Initializes the tour randomly by shuffling the city order.
	fn initial_tour(&mut self) {
		let mut rng = rand::thread_rng();
		self.tour = (0..self.tsp.dim()).collect();
		self.tour.shuffle(&mut rng);
		self.cost = self.calculate_tour_cost();
	}

	/// Calculates the total cost of the current tour.
	///
	/// # Returns
	/// The cost of the tour.
	fn calculate_tour_cost(&self) -> f64 {
		let mut cost = 0.0;
		for i in 0..self.tour.len() {
			let from = self.tour[i];
			let to = self.tour[(i + 1) % self.tour.len()];
			cost += self.cost(from, to);
		}
		cost
	}

	/// Attempts to improve the current tour.
	///
	/// # Returns
	/// `true` if the tour was improved, otherwise `false`.
	fn improve_tour(&mut self) -> bool {
		for i in 0..self.tour.len() {
			if self.improve_step(i) {
				return true;
			}
		}
		false
	}

	/// Performs a single step of the tour improvement process.
	///
	/// # Arguments
	/// * `start` - The starting point for the step.
	///
	/// # Returns
	/// `true` if the step resulted in an improved tour.
	fn improve_step(&mut self, start: usize) -> bool {
		let mut t = vec![start];
		let mut gain = 0.0;

		loop {
			if let Some((next, new_gain)) = self.find_next(&t, gain) {
				t.push(next);
				gain = new_gain;

				if gain > 0.0 && self.is_tour_feasible(&t) {
					self.apply_move(&t);
					return true;
				}

				if t.len() >= self.tour.len() - 1 {
					break;
				}
			} else {
				break;
			}
		}

		false
	}

	/// Finds the next best step in the improvement process.
	///
	/// # Arguments
	/// * `t` - The partial tour.
	/// * `current_gain` - The current gain.
	///
	/// # Returns
	/// An optional tuple with the next city and the updated gain.
	fn find_next(&self, t: &Vec<usize>, current_gain: f64) -> Option<(usize, f64)> {
		let mut best_next = None;
		let mut best_gain = current_gain;

		for i in 0..self.tour.len() {
			if !t.contains(&i) {
				let gain = self.calculate_gain(t, i);
				if gain > best_gain {
					best_gain = gain;
					best_next = Some(i);
				}
			}
		}

		best_next.map(|next| (next, best_gain))
	}

	/// Calculates the gain of adding the next city to the partial tour.
	///
	/// # Arguments
	/// * `t` - The partial tour.
	/// * `next` - The next city to consider.
	///
	/// # Returns
	/// The gain from adding the next city.
	fn calculate_gain(&self, t: &Vec<usize>, next: usize) -> f64 {
		let last = t[t.len() - 1];
		let first = t[0];
		let removed_edge = self.cost(last, self.tour[(last + 1) % self.tour.len()]);
		let added_edge = self.cost(last, next);
		let closing_edge = if t.len() == self.tour.len() - 1 {
			self.cost(next, first)
		} else {
			0.0
		};

		removed_edge - added_edge - closing_edge
	}

	/// Checks if the partial tour is feasible (i.e., forms a valid tour).
	///
	/// # Arguments
	/// * `t` - The partial tour.
	///
	/// # Returns
	/// `true` if the tour is feasible, otherwise `false`.
	fn is_tour_feasible(&self, t: &Vec<usize>) -> bool {
		t.len() == self.tour.len() && t[0] == t[t.len() - 1]
	}

	/// Applies the improvement move to the tour.
	///
	/// # Arguments
	/// * `t` - The sequence of cities to apply.
	fn apply_move(&mut self, t: &Vec<usize>) {
		let mut new_tour = vec![0; self.tour.len()];
		for i in 0..t.len() - 1 {
			new_tour[i] = self.tour[t[i]];
		}
		self.tour = new_tour;
		self.cost = self.calculate_tour_cost();
	}
}

impl TspSolver for LinKernighan {
	/// Solves the TSP using the Lin-Kernighan heuristic.
	///
	/// # Returns
	/// A `Solution` containing the best tour and its cost.
	fn solve(&mut self) -> Solution {
		let mut iterations = 0;
		while iterations < self.max_iterations {
			if !self.improve_tour() {
				break;
			}
			iterations += 1;
		}

		Solution {
			tour: self.tour.clone(),
			length: self.cost,
		}
	}

	/// Returns the current tour.
	///
	/// # Returns
	/// A vector representing the current tour.
	fn tour(&self) -> Vec<usize> {
		self.tour.clone()
	}

	/// Returns the cost between two cities.
	///
	/// # Arguments
	/// * `from` - The starting city.
	/// * `to` - The destination city.
	///
	/// # Returns
	/// The cost between the two cities.
	fn cost(&self, from: usize, to: usize) -> f64 {
		self.tsp.weight(from, to)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
	use crate::algorithms::TspSolver;
	use crate::TspBuilder;

	#[test]
	fn lin_kernighan_initial_tour_randomized() {
		let data = "
		NAME : example
		COMMENT : this is
		COMMENT : a simple example
		TYPE : TSP
		DIMENSION : 5
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 1.2 3.4
		  2 5.6 7.8
		  3 3.4 5.6
		  4 9.0 1.2
		  5 6.0 2.2
		EOF
		";
		let tsp = TspBuilder::parse_str(data).unwrap();
		let solver = LinKernighan::new(tsp);
		assert_eq!(solver.tour.len(), 5);
		assert!(solver.cost > 0.0);
	}

	#[test]
	fn lin_kernighan_improve_tour() {
		let data = "
		NAME : example
		COMMENT : this is
		COMMENT : a simple example
		TYPE : TSP
		DIMENSION : 5
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 1.2 3.4
		  2 5.6 7.8
		  3 3.4 5.6
		  4 9.0 1.2
		  5 6.0 2.2
		EOF
		";
		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut nn = NearestNeighbor::new(tsp.clone());
		let base_tour = nn.solve().tour;
		let mut solver = LinKernighan::with_options(tsp, base_tour, false, 1000);
		let initial_cost = solver.cost;
		solver.improve_tour();
		assert!(solver.cost <= initial_cost);
	}

	#[test]
	fn lin_kernighan_no_improvement() {
		let data = "
		NAME : example
		COMMENT : this is
		COMMENT : a simple example
		TYPE : TSP
		DIMENSION : 5
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 1.2 3.4
		  2 5.6 7.8
		  3 3.4 5.6
		  4 9.0 1.2
		  5 6.0 2.2
		EOF
		";
		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = LinKernighan::new(tsp);
		let improved = solver.improve_tour();
		assert!(!improved);
	}

	#[test]
	fn lin_kernighan_solve() {
		let data = "
		NAME : example
		COMMENT : this is
		COMMENT : a simple example
		TYPE : TSP
		DIMENSION : 5
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 1.2 3.4
		  2 5.6 7.8
		  3 3.4 5.6
		  4 9.0 1.2
		  5 6.0 2.2
		EOF
		";
		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = LinKernighan::new(tsp);
		let solution = solver.solve();
		assert_eq!(solution.tour.len(), 5);
		assert!(solution.length > 0.0);
	}
}