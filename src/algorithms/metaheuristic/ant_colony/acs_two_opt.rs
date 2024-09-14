use crate::{AntColonySystem, Solution, Tsp, TspSolver, TwoOpt};

/// This module implements the ACS2Opt algorithm, which combines the Ant Colony System (ACS)
/// with the 2-opt local search algorithm for solving the Traveling Salesman Problem (TSP).
pub struct ACS2Opt {
	tsp: Tsp,
	acs: AntColonySystem,
}

impl ACS2Opt {
	/// Creates a new instance of the ACS2Opt algorithm with the specified options.
	///
	/// # Arguments
	///
	/// * `tsp` - The TSP instance to solve.
	/// * `alpha` - The pheromone importance factor.
	/// * `beta` - The heuristic importance factor.
	/// * `rho` - The pheromone evaporation rate.
	/// * `q0` - The probability of exploitation versus exploration.
	/// * `num_ants` - The number of ants in the colony.
	/// * `max_iterations` - The maximum number of iterations to perform.
	/// * `candidate_list_size` - The size of the candidate list for local search.
	///
	/// # Returns
	///
	/// A new instance of the ACS2Opt algorithm.
	pub fn with_options(tsp: Tsp, alpha: f64, beta: f64, rho: f64, q0: f64, num_ants: usize, max_iterations: usize, candidate_list_size: usize) -> ACS2Opt {
		let acs = AntColonySystem::with_options(tsp.clone(), alpha, beta, rho, q0, num_ants, max_iterations, candidate_list_size);

		ACS2Opt {
			tsp,
			acs: acs,
		}
	}
}

impl TspSolver for ACS2Opt {
	/// Solves the TSP using the ACS2Opt algorithm.
	///
	/// # Returns
	///
	/// A `Solution` struct containing the tour and its total cost.
	fn solve(&mut self) -> Solution {
		let base_solution = self.acs.solve();
		TwoOpt::from(self.tsp.clone(), base_solution.tour, false).solve()
	}

	/// Returns the tour of the TSP solution.
	///
	/// # Returns
	///
	/// A vector of node indices representing the tour.
	fn tour(&self) -> Vec<usize> {
		todo!("Implement tour method for ACS2Opt")
	}

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
	fn cost(&self, from: usize, to: usize) -> f64 {
		self.tsp.weight(from, to)
	}

	/// Returns the name of the algorithm.
	///
	/// # Returns
	///
	/// A string representing the name of the algorithm.
	fn format_name(&self) -> String {
		format!("ACS2Opt")
	}
}


#[cfg(test)]
mod tests {
	use crate::algorithms::TspSolver;
	use crate::ant_colony::acs_two_opt::ACS2Opt;
	use crate::TspBuilder;

	#[test]
	fn solves_simple_tsp_with_acs2opt() {
		let data = "
		NAME : simple
		TYPE : TSP
		DIMENSION : 4
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		  2 0.0 1.0
		  3 1.0 1.0
		  4 1.0 0.0
		EOF
		";

		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 10, 1000, 15);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 4);
		assert!((solution.length - 4.0).abs() < f64::EPSILON);
	}

	#[test]
	fn handles_single_node_with_acs2opt() {
		let data = "
		NAME : single
		TYPE : TSP
		DIMENSION : 1
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		EOF
		";

		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 10, 1000, 15);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 1);
		assert!((solution.length - 0.0).abs() < f64::EPSILON);
	}

	#[test]
	fn handles_two_nodes_with_acs2opt() {
		let data = "
		NAME : two_nodes
		TYPE : TSP
		DIMENSION : 2
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		  2 1.0 0.0
		EOF
		";

		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 10, 1000, 15);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 2);
		assert!((solution.length - 2.0).abs() < f64::EPSILON);
	}

	#[test]
	fn handles_non_euclidean_distances_with_acs2opt() {
		let data = "
		NAME : non_euclidean
		TYPE : TSP
		DIMENSION : 3
		EDGE_WEIGHT_TYPE: EXPLICIT
		EDGE_WEIGHT_FORMAT: FULL_MATRIX
		EDGE_WEIGHT_SECTION
		  0 2 9
		  1 0 6
		  15 7 0
		EOF
		";

		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 10, 1000, 15);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 3);
		assert!((solution.length - 17.0).abs() < f64::EPSILON);
	}
}