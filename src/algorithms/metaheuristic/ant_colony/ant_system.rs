use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;

/// This module implements the Ant System (AS) algorithm for solving the Traveling Salesman Problem (TSP).
pub struct AntSystem {
	tsp: Tsp,
	pheromones: Vec<Vec<f64>>,
	best_tour: Vec<usize>,
	best_cost: f64,
	alpha: f64,
	beta: f64,
	rho: f64,
	num_ants: usize,
	max_iterations: usize,
}

impl AntSystem {
	/// Creates a new instance of the Ant System algorithm with the specified options.
	///
	/// # Arguments
	///
	/// * `tsp` - The TSP instance to solve.
	/// * `alpha` - The pheromone importance factor.
	/// * `beta` - The heuristic importance factor.
	/// * `rho` - The pheromone evaporation rate.
	/// * `num_ants` - The number of ants in the colony.
	/// * `max_iterations` - The maximum number of iterations to perform.
	///
	/// # Returns
	///
	/// A new instance of the Ant System algorithm.
	pub fn with_options(tsp: Tsp, alpha: f64, beta: f64, rho: f64, num_ants: usize, max_iterations: usize) -> AntSystem {
		let dim = tsp.dim();
		let initial_pheromone = 1.0 / (dim as f64);
		let pheromones = vec![vec![initial_pheromone; dim]; dim];

		AntSystem {
			tsp,
			pheromones,
			best_tour: vec![],
			best_cost: f64::INFINITY,
			alpha,
			beta,
			rho,
			num_ants,
			max_iterations,
		}
	}

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

	/// Constructs a solution (tour) for the TSP using the Ant System algorithm.
	///
	/// # Returns
	///
	/// A vector of node indices representing the constructed tour.
	fn construct_solution(&self) -> Vec<usize> {
		let mut rng = rand::thread_rng();
		let mut tour = vec![0; self.tsp.dim()];
		let mut visited = vec![false; self.tsp.dim()];

		tour[0] = rng.gen_range(0..self.tsp.dim());
		visited[tour[0]] = true;

		for i in 1..self.tsp.dim() {
			tour[i] = self.select_next_city(&tour[0..i], &visited, &mut rng);
			visited[tour[i]] = true;
		}

		tour
	}

	/// Selects the next city to visit in the tour based on pheromone levels and heuristic information.
	///
	/// # Arguments
	///
	/// * `partial_tour` - A slice of node indices representing the partial tour constructed so far.
	/// * `visited` - A vector indicating whether each city has been visited.
	/// * `rng` - A random number generator.
	///
	/// # Returns
	///
	/// The index of the next city to visit.
	fn select_next_city(&self, partial_tour: &[usize], visited: &[bool], rng: &mut ThreadRng) -> usize {
		let current_city = partial_tour[partial_tour.len() - 1];
		let mut probabilities = vec![0.0; self.tsp.dim()];
		let mut total = 0.0;

		for (city, &visited) in visited.iter().enumerate() {
			if !visited {
				let pheromone = self.pheromones[current_city][city];
				let distance = 1.0 / self.tsp.weight(current_city, city);
				let probability = pheromone.powf(self.alpha) * distance.powf(self.beta);
				probabilities[city] = probability;
				total += probability;
			}
		}

		let random_value = rng.gen::<f64>() * total;
		let mut cumulative = 0.0;

		for (city, &probability) in probabilities.iter().enumerate() {
			cumulative += probability;
			if cumulative >= random_value {
				return city;
			}
		}

		visited.iter().position(|&v| !v).unwrap()
	}

	/// Updates the pheromone levels based on the solutions found by the ants.
	///
	/// # Arguments
	///
	/// * `solutions` - A slice of vectors, each representing a tour found by an ant.
	fn update_pheromones(&mut self, solutions: &[Vec<usize>]) {
		// Evaporation
		for row in self.pheromones.iter_mut() {
			for pheromone in row.iter_mut() {
				*pheromone *= 1.0 - self.rho;
			}
		}

		// Deposit
		for solution in solutions {
			let cost = self.calculate_tour_cost(solution);
			let deposit = 1.0 / cost;

			for i in 0..solution.len() {
				let from = solution[i];
				let to = solution[(i + 1) % solution.len()];
				self.pheromones[from][to] += deposit;
				self.pheromones[to][from] += deposit;
			}
		}
	}

	/// Updates the best solution found so far based on the solutions found by the ants.
	///
	/// # Arguments
	///
	/// * `solutions` - A slice of vectors, each representing a tour found by an ant.
	fn update_best_solution(&mut self, solutions: &[Vec<usize>]) {
		for solution in solutions {
			let cost = self.calculate_tour_cost(solution);
			if cost < self.best_cost {
				self.best_tour = solution.clone();
				self.best_cost = cost;
			}
		}
	}
}

impl TspSolver for AntSystem {
	/// Solves the TSP using the Ant System algorithm.
	///
	/// # Returns
	///
	/// A `Solution` struct containing the tour and its total cost.
	fn solve(&mut self) -> Solution {
		for _ in 0..self.max_iterations {
			let mut solutions = Vec::with_capacity(self.num_ants);

			for _ in 0..self.num_ants {
				let solution = self.construct_solution();
				solutions.push(solution);
			}

			self.update_pheromones(&solutions);
			self.update_best_solution(&solutions);
		}

		Solution {
			tour: self.best_tour.clone(),
			length: self.best_cost,
		}
	}

	/// Returns the tour of the TSP solution.
	///
	/// # Returns
	///
	/// A vector of node indices representing the tour.
	fn tour(&self) -> Vec<usize> {
		self.best_tour.clone()
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
		format!("AS")
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::TspBuilder;

	#[test]
	fn solves_simple_tsp_with_ant_system() {
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
		let mut solver = AntSystem::with_options(tsp, 0.1, 2.0, 0.1, 10, 1000);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 4);
		assert!((solution.length - 4.0).abs() < f64::EPSILON);
	}

	#[test]
	fn handles_single_node_with_ant_system() {
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
		let mut solver = AntSystem::with_options(tsp, 0.1, 2.0, 0.1, 10, 1000);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 1);
		assert!((solution.length - 0.0).abs() < f64::EPSILON);
	}

	#[test]
	fn handles_two_nodes_with_ant_system() {
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
		let mut solver = AntSystem::with_options(tsp, 0.1, 2.0, 0.1, 10, 1000);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 2);
		assert!((solution.length - 2.0).abs() < f64::EPSILON);
	}

	#[test]
	fn handles_non_euclidean_distances_with_ant_system() {
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
		let mut solver = AntSystem::with_options(tsp, 0.1, 2.0, 0.1, 10, 1000);
		let solution = solver.solve();

		assert_eq!(solution.tour.len(), 3);
		assert!((solution.length - 17.0).abs() < f64::EPSILON);
	}
}