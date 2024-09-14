use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;


/// A struct representing the 3-opt algorithm for solving the Traveling Salesman Problem (TSP).
///
/// # Fields
///
/// * `tsp` - A reference to the TSP instance.
/// * `tour` - A vector representing the current tour.
/// * `cost` - The cost of the current tour.
/// * `verbose` - A flag indicating whether to print verbose output.
/// * `base_tour` - The initial tour based on the node coordinates.
pub struct ThreeOpt<'a> {
	tsp: &'a Tsp,
	tour: Vec<usize>,
	cost: f64,
	verbose: bool,
	base_tour: Vec<usize>,
}

impl ThreeOpt<'_> {
	/// Creates a new instance of the ThreeOpt struct.
	///
	/// # Arguments
	///
	/// * `tsp` - A reference to the TSP instance.
	pub fn new<'a>(tsp: &'a Tsp) -> ThreeOpt<'a> {
		ThreeOpt {
			tsp,
			tour: vec![],
			cost: 0.0,
			verbose: false,
			base_tour: tsp.node_coords().iter().map(|node| *node.0).collect(),
		}
	}

	/// Optimizes the current tour using the 3-opt algorithm.
	///
	/// This method iteratively improves the tour by considering all possible
	/// 3-opt moves and selecting the one that results in the greatest reduction
	/// in tour cost. The process continues until no further improvement is possible.
	pub fn optimize(&mut self) {
		let n = self.tour.len();
		let mut improved = true;

		if n <= 3 {
			return;
		}

		while improved {
			improved = false;
			for i in 0..n - 2 {
				for j in i + 2..n - 1 {
					for k in j + 2..n + (i > 0) as usize {
						if k >= n { continue; } // Ensure k stays in bounds

						let new_tours = self.generate_new_tours(i, j, k);

						for new_tour in new_tours {
							let new_cost = self.calculate_tour_cost_with(&new_tour);

							if new_cost < self.cost {
								self.tour = new_tour;
								self.cost = new_cost;
								improved = true;

								if self.verbose {
									println!(
										"3OPT: Improved tour at segments ({}, {}, {}), new cost: {}",
										i, j, k, self.cost
									);
								}
							}
						}
					}
				}
			}
		}
	}

	/// Generates new tours by performing 2-opt and 3-opt moves.
	///
	/// # Arguments
	///
	/// * `i` - The first index for the 3-opt move.
	/// * `j` - The second index for the 3-opt move.
	/// * `k` - The third index for the 3-opt move.
	fn generate_new_tours(&self, i: usize, j: usize, k: usize) -> Vec<Vec<usize>> {
		let mut new_tours = Vec::new();
		let _n = self.tour.len();

		let mut tour1 = self.tour.clone();
		let mut tour2 = self.tour.clone();
		let mut tour3 = self.tour.clone();
		let mut tour4 = self.tour.clone();
		let mut tour5 = self.tour.clone();
		let mut tour6 = self.tour.clone();
		let mut tour7 = self.tour.clone();

		// Case 1: no change (do nothing)

		// Case 2: 2-opt between i+1 and j (reverse segment between i+1 and j)
		tour1[i + 1..=j].reverse();

		// Case 3: 2-opt between j+1 and k (reverse segment between j+1 and k)
		tour2[j + 1..=k].reverse();

		// Case 4: 2-opt between i+1 and k (reverse segment between i+1 and k)
		tour3[i + 1..=k].reverse();

		// Case 5: 3-opt with reversing segments i+1 to j and j+1 to k
		tour4[i + 1..=j].reverse();
		tour4[j + 1..=k].reverse();

		// Case 6: 3-opt with reversing segments i+1 to j and i+1 to k
		tour5[i + 1..=j].reverse();
		tour5[i + 1..=k].reverse();

		// Case 7: 3-opt with reversing segments j+1 to k and i+1 to k
		tour6[j + 1..=k].reverse();
		tour6[i + 1..=k].reverse();

		// Case 8: 3-opt with reversing all segments
		tour7[i + 1..=j].reverse();
		tour7[j + 1..=k].reverse();
		tour7[i + 1..=k].reverse();

		new_tours.push(tour1);
		new_tours.push(tour2);
		new_tours.push(tour3);
		new_tours.push(tour4);
		new_tours.push(tour5);
		new_tours.push(tour6);
		new_tours.push(tour7);

		new_tours
	}

	/// Calculates the cost of a given tour.
	///
	/// # Arguments
	///
	/// * `tour` - A reference to a vector of node indices representing the tour.
	///
	/// # Returns
	///
	/// The total cost of the tour as a `f64`.
	fn calculate_tour_cost_with(&self, tour: &Vec<usize>) -> f64 {
		let mut cost = 0.0;
		let len = tour.len();
		for i in 0..len {
			let from = tour[i];
			let to = tour[(i + 1) % len];
			cost += self.tsp.weight(from, to) as f64;
		}
		cost
	}
}

impl TspSolver for ThreeOpt<'_> {
	/// Solves the TSP using the 3-opt algorithm.
	///
	/// This method initializes the tour and cost, prints the initial state if verbose is enabled,
	/// and then calls the `optimize` method to improve the tour.
	///
	/// # Returns
	///
	/// A `Solution` struct containing the optimized tour and its total cost.
	fn solve(&mut self) -> Solution {
		self.tour = self.base_tour.clone();
		self.cost = self.calculate_tour_cost(&self.base_tour);

		if self.verbose {
			println!("Initial tour: {:?}", self.tour);
			println!("Initial cost: {}", self.cost);
		}

		self.optimize();

		Solution {
			tour: self.tour.clone(),
			length: self.cost,
		}
	}
	fn tour(&self) -> Vec<usize> {
		self.tour.clone()
	}

	fn cost(&self, from: usize, to: usize) -> f64 {
		self.tsp.weight(from, to)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::algorithms::TspSolver;
	use crate::TspBuilder;
	#[test]
	fn three_opt_initial_tour_cost() {
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
		let solver = ThreeOpt::new(&tsp);
		let initial_cost = solver.calculate_tour_cost_with(&solver.base_tour);
		assert_eq!(solver.cost, 0.0);
		assert!(initial_cost > 0.0);
	}

	#[test]
	fn three_opt_optimization_reduces_cost() {
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
		let mut solver = ThreeOpt::new(&tsp);
		solver.solve();
		assert!(solver.cost < solver.calculate_tour_cost_with(&solver.base_tour));
	}

	#[test]
	fn three_opt_handles_empty_tour() {
		let data = "
        NAME : example
        COMMENT : this is
        COMMENT : a simple example
        TYPE : TSP
        DIMENSION : 0
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
        EOF
        ";
		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = ThreeOpt::new(&tsp);
		let solution = solver.solve();
		assert_eq!(solution.tour.len(), 0);
		assert_eq!(solution.length, 0.0);
	}

	#[test]
	fn three_opt_handles_single_node_tour() {
		let data = "
        NAME : example
        COMMENT : this is
        COMMENT : a simple example
        TYPE : TSP
        DIMENSION : 1
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 1.2 3.4
        EOF
        ";
		let tsp = TspBuilder::parse_str(data).unwrap();
		let mut solver = ThreeOpt::new(&tsp);
		let solution = solver.solve();
		assert_eq!(solution.tour.len(), 1);
		assert_eq!(solution.length, 0.0);
	}
}
