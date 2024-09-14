use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::cmp::Ordering;
use std::f64;

/// A genetic algorithm implementation for solving the Traveling Salesman Problem (TSP).
///
/// This struct uses genetic algorithm principles to find an approximate solution to the TSP.
/// It maintains a population of potential solutions (tours) and evolves them over generations.
///
/// # Examples
///
/// ```
/// use ibn_battuta::{GeneticAlgorithm, TspBuilder, TspSolver};
///
/// let tsp_data = "
///     NAME : example
///     DIMENSION : 5
///     EDGE_WEIGHT_TYPE: EUC_2D
///     NODE_COORD_SECTION
///       1 1.2 3.4
///       2 5.6 7.8
///       3 3.4 5.6
///       4 9.0 1.2
///       5 6.0 2.2
///     EOF
/// ";
/// let tsp = TspBuilder::parse_str(tsp_data).unwrap();
/// let mut solver = GeneticAlgorithm::with_options(tsp, 100, 5, 0.7, 0.01, 500);
/// let solution = solver.solve();
///
/// assert_eq!(solution.tour.len(), 5);
/// ```
pub struct GeneticAlgorithm {
	tsp: Tsp,
	population: Vec<Vec<usize>>,
	population_size: usize,
	elite_size: usize,
	crossover_rate: f64,
	mutation_rate: f64,
	max_generations: usize,
}

impl GeneticAlgorithm {
	/// Creates a new `GeneticAlgorithm` instance with the specified parameters.
	///
	/// # Arguments
	///
	/// * `tsp` - The TSP instance to solve.
	/// * `population_size` - The size of the population in each generation.
	/// * `elite_size` - The number of best individuals to carry over to the next generation.
	/// * `crossover_rate` - The probability of performing crossover between two parents.
	/// * `mutation_rate` - The probability of mutating an individual.
	/// * `max_generations` - The maximum number of generations to evolve.
	///
	/// # Examples
	///
	/// ```
	/// use ibn_battuta::{GeneticAlgorithm, TspBuilder};
	///
	/// let tsp = TspBuilder::parse_str("NAME : example\nDIMENSION : 5\n...").unwrap();
	/// let ga = GeneticAlgorithm::with_options(tsp, 100, 5, 0.7, 0.01, 500);
	/// ```
	pub fn with_options(
		tsp: Tsp,
		population_size: usize,
		elite_size: usize,
		crossover_rate: f64,
		mutation_rate: f64,
		max_generations: usize,
	) -> GeneticAlgorithm {
		let population_size = population_size.max(2);
		let elite_size = elite_size.min(population_size / 2);

		let mut ga = GeneticAlgorithm {
			tsp,
			population: Vec::with_capacity(population_size),
			population_size,
			elite_size,
			crossover_rate,
			mutation_rate,
			max_generations,
		};
		ga.initialize_population();
		ga
	}
	/// Initializes the population with random tours and applies a greedy initialization.
	fn initialize_population(&mut self) {
		let mut rng = rand::thread_rng();
		for _ in 0..self.population_size {
			let mut tour: Vec<usize> = (0..self.tsp.dim()).collect();
			tour.shuffle(&mut rng);
			self.population.push(tour);
		}
		self.greedy_init();
	}

	fn greedy_init(&mut self) {
		let num_city = self.tsp.dim();
		let mut start_index = 0;
		while self.population.len() < self.population_size {
			let mut rest: Vec<usize> = (0..num_city).collect();
			if start_index >= num_city {
				start_index = rand::thread_rng().gen_range(0..num_city);
				self.population.push(self.population[start_index].clone());
				continue;
			}
			let mut current = start_index;
			rest.retain(|&x| x != current);
			let mut result_one = vec![current];
			while !rest.is_empty() {
				let (tmp_choose, _) = rest.iter()
										  .map(|&x| (x, self.cost(current, x)))
										  .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
										  .unwrap();
				current = tmp_choose;
				result_one.push(tmp_choose);
				rest.retain(|&x| x != tmp_choose);
			}
			self.population.push(result_one);
			start_index += 1;
		}
	}

	/// Calculates the fitness of a given tour.
	///
	/// The fitness is the inverse of the tour cost, so lower cost tours have higher fitness.
	///
	/// # Arguments
	///
	/// * `tour` - A slice representing a tour of cities.
	///
	/// # Returns
	///
	/// The fitness value of the tour.
	fn calculate_fitness(&self, tour: &[usize]) -> f64 {
		1.0 / self.calculate_tour_cost(tour)
	}

	/// Calculates the total cost of a given tour.
	///
	/// # Arguments
	///
	/// * `tour` - A slice representing a tour of cities.
	///
	/// # Returns
	///
	/// The total cost of the tour.
	fn calculate_tour_cost(&self, tour: &[usize]) -> f64 {
		tour.windows(2)
			.map(|w| self.cost(w[0], w[1]))
			.sum::<f64>()
			+ self.cost(*tour.last().unwrap(), tour[0])
	}

	fn select_parents(&self, fitnesses: &[f64]) -> Vec<usize> {
		let mut rng = rand::thread_rng();
		let total_fitness: f64 = fitnesses.iter().sum();
		let mut selected = Vec::with_capacity(self.population_size);

		for _ in 0..self.population_size {
			let mut r = rng.gen::<f64>() * total_fitness;
			for (i, &fitness) in fitnesses.iter().enumerate() {
				r -= fitness;
				if r <= 0.0 {
					selected.push(i);
					break;
				}
			}
		}

		selected
	}

	/// Performs crossover between two parent tours to produce a child tour.
	///
	/// # Arguments
	///
	/// * `parent1` - A slice representing the first parent tour.
	/// * `parent2` - A slice representing the second parent tour.
	///
	/// # Returns
	///
	/// A new tour resulting from the crossover of the two parent tours.
	fn crossover(&self, parent1: &[usize], parent2: &[usize]) -> Vec<usize> {
		let mut rng = rand::thread_rng();
		let start = rng.gen_range(0..parent1.len());
		let end = rng.gen_range(start..parent1.len());

		let mut child = vec![0; parent1.len()];
		child[start..=end].copy_from_slice(&parent1[start..=end]);

		let mut j = (end + 1) % parent1.len();
		for &city in parent2.iter().chain(parent2.iter()) {
			if !child[start..=end].contains(&city) {
				child[j] = city;
				j = (j + 1) % parent1.len();
				if j == start {
					break;
				}
			}
		}

		child
	}

	/// Mutates a tour by swapping two random cities.
	///
	/// # Arguments
	///
	/// * `tour` - A mutable reference to the tour to be mutated.
	fn mutate(&self, tour: &mut Vec<usize>) {
		let mut rng = rand::thread_rng();
		if rng.gen::<f64>() < self.mutation_rate {
			let i = rng.gen_range(0..tour.len());
			let j = rng.gen_range(0..tour.len());
			tour.swap(i, j);
		}
	}

	/// Evolves the population for one generation.
	///
	/// # Returns
	///
	/// The best tour found in the current generation.
	fn evolve(&mut self) -> Vec<usize> {
		let fitnesses: Vec<f64> = self.population.iter()
									  .map(|tour| self.calculate_fitness(tour))
									  .collect();

		let mut next_generation = Vec::with_capacity(self.population_size);

		// Elitism
		let mut indexed_fitnesses: Vec<(usize, f64)> = fitnesses.iter().enumerate()
																.map(|(i, &f)| (i, f))
																.collect();
		indexed_fitnesses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
		for &(index, _) in indexed_fitnesses.iter().take(self.elite_size) {
			next_generation.push(self.population[index].clone());
		}

		let parents = self.select_parents(&fitnesses);

		while next_generation.len() < self.population_size {
			let parent1 = &self.population[parents[rand::thread_rng().gen_range(0..parents.len())]];
			let parent2 = &self.population[parents[rand::thread_rng().gen_range(0..parents.len())]];

			let mut child = if rand::thread_rng().gen::<f64>() < self.crossover_rate {
				self.crossover(parent1, parent2)
			} else {
				parent1.clone()
			};

			self.mutate(&mut child);
			next_generation.push(child);
		}

		self.population = next_generation;

		self.population[indexed_fitnesses[0].0].clone()
	}
}

impl TspSolver for GeneticAlgorithm {
	fn solve(&mut self) -> Solution {
		let mut best_tour = Vec::new();
		let mut best_cost = f64::INFINITY;

		for _ in 0..self.max_generations {
			let current_best = self.evolve();
			let current_cost = self.calculate_tour_cost(&current_best);

			if current_cost < best_cost {
				best_cost = current_cost;
				best_tour = current_best;
			}
		}

		Solution {
			tour: best_tour,
			length: best_cost,
		}
	}

	fn tour(&self) -> Vec<usize> {
		self.population[0].clone()
	}

	fn cost(&self, from: usize, to: usize) -> f64 {
		self.tsp.weight(from, to)
	}

	fn format_name(&self) -> String {
		format!("GA")
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::TspBuilder;

	#[test]
	fn test_genetic_algorithm_creation() {
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
		let ga = GeneticAlgorithm::with_options(tsp, 50, 5, 0.7, 0.01, 100);

		assert_eq!(ga.population_size, 50);
		assert_eq!(ga.elite_size, 5);
		assert_eq!(ga.crossover_rate, 0.7);
		assert_eq!(ga.mutation_rate, 0.01);
		assert_eq!(ga.max_generations, 100);
		assert_eq!(ga.population.len(), 50);
	}

	#[test]
	fn test_tour_cost_calculation() {
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
		let ga = GeneticAlgorithm::with_options(tsp.clone(), 10, 2, 0.7, 0.01, 100);

		let tour = vec![0, 1, 2, 3];
		let cost = ga.calculate_tour_cost(&tour);

		let expected_cost = tsp.weight(0, 1) + tsp.weight(1, 2) + tsp.weight(2, 3) + tsp.weight(3, 0);
		// Expected cost: 3 + 4 + 3 + 4 = 14
		assert!((cost - expected_cost).abs() < 1e-6);
	}

	#[test]
	fn test_crossover() {
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
		let ga = GeneticAlgorithm::with_options(tsp, 10, 2, 0.7, 0.01, 100);

		let parent1 = vec![0, 1, 2, 3, 4];
		let parent2 = vec![4, 3, 2, 1, 0];

		let child = ga.crossover(&parent1, &parent2);

		assert_eq!(child.len(), 5);
		assert!(child.iter().all(|&x| x < 5));
		assert_eq!(child.iter().collect::<std::collections::HashSet<_>>().len(), 5);
	}

	// TODO: Add more tests
}