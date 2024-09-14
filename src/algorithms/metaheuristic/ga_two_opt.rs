use crate::{GeneticAlgorithm, Solution, Tsp, TspSolver, TwoOpt};

/// GA2Opt is a hybrid algorithm that combines Genetic Algorithm (GA) with 2-Opt local search
/// to solve the Traveling Salesman Problem (TSP).
pub struct GA2Opt {
	tsp: Tsp,
	ga: GeneticAlgorithm,
}

impl GA2Opt {
	/// Creates a new GA2Opt solver with the specified options.
	///
	/// # Arguments
	///
	/// * `tsp` - The TSP instance to solve
	/// * `population_size` - The size of the population in the genetic algorithm
	/// * `elite_size` - The number of elite solutions to keep in each generation
	/// * `crossover_rate` - The probability of crossover occurring
	/// * `mutation_rate` - The probability of mutation occurring
	/// * `max_generations` - The maximum number of generations to run
	///
	/// # Examples
	///
	/// ```
	/// use ibn_battuta::{Tsp, TspBuilder, GA2Opt};
	///
	/// let tsp = TspBuilder::parse_path("path/to/tsp/file.tsp").unwrap();
	/// let solver = GA2Opt::with_options(tsp, 100, 5, 0.7, 0.01, 500);
	/// ```
	pub fn with_options(
		tsp: Tsp,
		population_size: usize,
		elite_size: usize,
		crossover_rate: f64,
		mutation_rate: f64,
		max_generations: usize,
	) -> GA2Opt {
		let acs = GeneticAlgorithm::with_options(
			tsp.clone(),
			population_size,
			elite_size,
			crossover_rate,
			mutation_rate,
			max_generations,
		);

		GA2Opt { tsp, ga: acs }
	}
}

impl TspSolver for GA2Opt {
	/// Solves the TSP using the GA2Opt algorithm.
	///
	/// This method first uses the genetic algorithm to find a base solution,
	/// then applies the 2-Opt local search to further improve the solution.
	///
	/// # Returns
	///
	/// A `Solution` containing the best tour found and its cost.
	///
	/// # Examples
	///
	/// ```
	/// use ibn_battuta::{ GA2Opt, TspBuilder, TspSolver};
	///
	/// let tsp = TspBuilder::parse_path("path/to/tsp/file.tsp").unwrap();
	/// let mut solver = GA2Opt::with_options(tsp, 100, 5, 0.7, 0.01, 500);
	/// let solution = solver.solve();
	/// println!("Best tour cost: {}", solution.length);
	/// ```
	fn solve(&mut self) -> Solution {
		let base_solution = self.ga.solve();
		TwoOpt::from(self.tsp.clone(), base_solution.tour, false).solve()
	}

	/// Returns the current best tour found by the solver.
	///
	/// # Returns
	///
	/// A vector of city indices representing the tour.
	fn tour(&self) -> Vec<usize> {
		self.ga.tour()
	}

	/// Calculates the cost between two cities in the TSP.
	///
	/// # Arguments
	///
	/// * `from` - The index of the starting city
	/// * `to` - The index of the ending city
	///
	/// # Returns
	///
	/// The cost (or distance) between the two cities.
	fn cost(&self, from: usize, to: usize) -> f64 {
		self.tsp.weight(from, to)
	}

	/// Returns the name of the solver.
	///
	/// # Returns
	///
	/// A string representation of the solver's name.
	fn format_name(&self) -> String {
		format!("GA2Opt")
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{NearestNeighbor, TspBuilder};

	#[test]
	fn test_gr17() {
		let path = "data/tsplib/gr17.tsp";
		let tsp = TspBuilder::parse_path(path).unwrap();

		test_instance(tsp);
	}

	fn test_instance(tsp: Tsp) {
		let size = tsp.dim();

		let mut solver = GA2Opt::with_options(tsp.clone(), 100, 5, 0.7, 0.01, 500);

		let solution = solver.solve();

		assert_eq!(solution.tour.len(), size);
		assert!(solution.length > 0.0, "Solution cost should be positive");

		let nn_solution = NearestNeighbor::new(tsp.clone()).solve();
		let nn_cost = nn_solution.length;

		assert!(solution.length <= nn_cost, "GA2Opt should find a solution with cost <= NN");
	}
	#[test]
	fn test_p43() {
		let path = "data/tsplib/berlin52.tsp";
		let tsp = TspBuilder::parse_path(path).unwrap();
		test_instance(tsp);
	}

	#[test]
	fn test_tour_and_cost() {
		let path = "data/tsplib/gr17.tsp";
		let tsp = TspBuilder::parse_path(path).unwrap();
		let mut solver = GA2Opt::with_options(tsp.clone(), 100, 5, 0.7, 0.01, 500);

		solver.solve(); // Run the solver

		let tour = solver.tour();
		assert_eq!(tour.len(), tsp.dim(), "Tour length should match TSP dimension");

		// Check if the tour is a valid permutation
		let mut sorted_tour = tour.clone();
		sorted_tour.sort();
		assert_eq!(
			sorted_tour,
			(0..tsp.dim()).collect::<Vec<usize>>(),
			"Tour should be a valid permutation"
		);

		// Test cost calculation
		let cost = solver.cost(0, 1);
		assert!(cost > 0.0, "Cost between cities should be positive");
	}

	#[test]
	fn test_format_name() {
		let path = "data/tsplib/gr17.tsp";
		let tsp = TspBuilder::parse_path(path).unwrap();
		let solver = GA2Opt::with_options(tsp, 100, 5, 0.7, 0.01, 500);

		assert_eq!(solver.format_name(), "GA2Opt", "Solver name should be GA2Opt");
	}
}