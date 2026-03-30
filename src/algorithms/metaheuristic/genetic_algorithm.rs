use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use rand::rngs::StdRng;
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
/// use ibn_battuta::experimental::GeneticAlgorithm;
/// use ibn_battuta::{TspBuilder, TspSolver};
///
/// let tsp_data = "
///     NAME : example
///     TYPE : TSP
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
    population_costs: Vec<f64>,
    population_size: usize,
    elite_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    max_generations: usize,
    rng: StdRng,
    seed: u64,
}

impl GeneticAlgorithm {
    const TOURNAMENT_SIZE: usize = 5;
    const GREEDY_SEED_RATIO: f64 = 0.5;

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
    /// use ibn_battuta::experimental::GeneticAlgorithm;
    /// use ibn_battuta::TspBuilder;
    ///
    /// let tsp = TspBuilder::parse_str("
    /// NAME : example
    /// TYPE : TSP
    /// DIMENSION : 3
    /// EDGE_WEIGHT_TYPE : EUC_2D
    /// NODE_COORD_SECTION
    /// 1 0 0
    /// 2 1 0
    /// 3 0 1
    /// EOF
    /// ").unwrap();
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
        Self::with_options_and_seed(
            tsp,
            population_size,
            elite_size,
            crossover_rate,
            mutation_rate,
            max_generations,
            rand::random(),
        )
    }

    pub fn with_options_and_seed(
        tsp: Tsp,
        population_size: usize,
        elite_size: usize,
        crossover_rate: f64,
        mutation_rate: f64,
        max_generations: usize,
        seed: u64,
    ) -> GeneticAlgorithm {
        let population_size = population_size.max(2);
        let elite_size = elite_size.min(population_size / 2);

        let mut ga = GeneticAlgorithm {
            tsp,
            population: Vec::with_capacity(population_size),
            population_costs: Vec::with_capacity(population_size),
            population_size,
            elite_size,
            crossover_rate,
            mutation_rate,
            max_generations,
            rng: StdRng::seed_from_u64(seed),
            seed,
        };
        ga.initialize_population();
        ga
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Initializes the population with random tours and applies a greedy initialization.
    fn initialize_population(&mut self) {
        let greedy_count = ((self.population_size as f64 * Self::GREEDY_SEED_RATIO).round()
            as usize)
            .max(1)
            .min(self.tsp.dim());

        for start_index in 0..greedy_count {
            let (tour, cost) = self.build_greedy_tour(start_index);
            self.population.push(tour);
            self.population_costs.push(cost);
        }

        while self.population.len() < self.population_size {
            let (tour, cost) = self.random_tour();
            self.population.push(tour);
            self.population_costs.push(cost);
        }
    }

    fn random_tour(&mut self) -> (Vec<usize>, f64) {
        let mut tour: Vec<usize> = (0..self.tsp.dim()).collect();
        tour.shuffle(&mut self.rng);
        let cost = self.calculate_tour_cost(&tour);
        (tour, cost)
    }

    fn build_greedy_tour(&self, start_index: usize) -> (Vec<usize>, f64) {
        let num_city = self.tsp.dim();
        let mut rest: Vec<usize> = (0..num_city).collect();
        let mut current = start_index;
        rest.retain(|&x| x != current);
        let mut result_one = vec![current];
        while !rest.is_empty() {
            let (tmp_choose, _) = rest
                .iter()
                .map(|&x| (x, self.cost(current, x)))
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
                .unwrap();
            current = tmp_choose;
            result_one.push(tmp_choose);
            rest.retain(|&x| x != tmp_choose);
        }
        let result_cost = self.calculate_tour_cost(&result_one);
        (result_one, result_cost)
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
        tour.windows(2).map(|w| self.cost(w[0], w[1])).sum::<f64>()
            + self.cost(*tour.last().unwrap(), tour[0])
    }

    fn select_parents(&mut self) -> Vec<usize> {
        let mut selected = Vec::with_capacity(self.population_size);
        let tournament_size = Self::TOURNAMENT_SIZE.min(self.population.len()).max(1);

        for _ in 0..self.population_size {
            let mut chosen = self.rng.gen_range(0..self.population.len());
            let mut chosen_cost = self.population_costs[chosen];
            for _ in 1..tournament_size {
                let challenger = self.rng.gen_range(0..self.population.len());
                let challenger_cost = self.population_costs[challenger];
                if challenger_cost < chosen_cost {
                    chosen = challenger;
                    chosen_cost = challenger_cost;
                }
            }
            selected.push(chosen);
        }

        selected
    }

    fn choose_parent_pair(&mut self, parents: &[usize]) -> (usize, usize) {
        let parent1_idx = parents[self.rng.gen_range(0..parents.len())];
        let mut parent2_idx = parents[self.rng.gen_range(0..parents.len())];

        for _ in 0..4 {
            if parent1_idx != parent2_idx {
                break;
            }
            parent2_idx = parents[self.rng.gen_range(0..parents.len())];
        }

        if parent1_idx == parent2_idx && self.population.len() > 1 {
            parent2_idx = (parent1_idx + 1) % self.population.len();
        }

        (parent1_idx, parent2_idx)
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
    fn crossover_with_rng(parent1: &[usize], parent2: &[usize], rng: &mut StdRng) -> Vec<usize> {
        let start = rng.gen_range(0..parent1.len());
        let end = rng.gen_range(start..parent1.len());

        let mut child = vec![0; parent1.len()];
        let mut used = vec![false; parent1.len()];
        child[start..=end].copy_from_slice(&parent1[start..=end]);
        for &city in &parent1[start..=end] {
            used[city] = true;
        }

        let mut j = (end + 1) % parent1.len();
        for &city in parent2.iter().chain(parent2.iter()) {
            if !used[city] {
                child[j] = city;
                used[city] = true;
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
    fn mutate_with_rng(tour: &mut [usize], mutation_rate: f64, rng: &mut StdRng) {
        if rng.gen::<f64>() < mutation_rate {
            let i = rng.gen_range(0..tour.len());
            let j = rng.gen_range(0..tour.len());
            if i != j {
                tour.swap(i, j);
            }
        }
    }

    /// Evolves the population for one generation.
    ///
    /// # Returns
    ///
    /// The best tour found in the current generation.
    fn evolve(&mut self) -> (Vec<usize>, f64) {
        let mut next_generation = Vec::with_capacity(self.population_size);
        let mut next_generation_costs = Vec::with_capacity(self.population_size);
        let mut generation_best_tour = Vec::new();
        let mut generation_best_cost = f64::INFINITY;

        // Elitism
        let mut indexed_costs: Vec<(usize, f64)> = self
            .population_costs
            .iter()
            .enumerate()
            .map(|(i, &cost)| (i, cost))
            .collect();
        indexed_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        for &(index, _) in indexed_costs.iter().take(self.elite_size) {
            let elite = self.population[index].clone();
            let elite_cost = self.population_costs[index];
            if elite_cost < generation_best_cost {
                generation_best_cost = elite_cost;
                generation_best_tour = elite.clone();
            }
            next_generation.push(elite);
            next_generation_costs.push(elite_cost);
        }

        let parents = self.select_parents();

        while next_generation.len() < self.population_size {
            let (parent1_idx, parent2_idx) = self.choose_parent_pair(&parents);
            let should_crossover = self.rng.gen::<f64>() < self.crossover_rate;
            let population = &self.population;
            let rng = &mut self.rng;

            let mut child = if should_crossover {
                Self::crossover_with_rng(&population[parent1_idx], &population[parent2_idx], rng)
            } else {
                population[parent1_idx].clone()
            };

            Self::mutate_with_rng(&mut child, self.mutation_rate, rng);
            let child_cost = self.calculate_tour_cost(&child);
            if child_cost < generation_best_cost {
                generation_best_cost = child_cost;
                generation_best_tour = child.clone();
            }
            next_generation.push(child);
            next_generation_costs.push(child_cost);
        }

        self.population = next_generation;
        self.population_costs = next_generation_costs;

        (generation_best_tour, generation_best_cost)
    }
}

impl TspSolver for GeneticAlgorithm {
    fn solve(&mut self) -> Solution {
        let mut best_tour = Vec::new();
        let mut best_cost = f64::INFINITY;

        for _ in 0..self.max_generations {
            let (current_best, current_cost) = self.evolve();

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
        self.population
            .iter()
            .zip(self.population_costs.iter())
            .min_by(|(_, cost_a), (_, cost_b)| {
                cost_a.partial_cmp(cost_b).unwrap_or(Ordering::Equal)
            })
            .map(|(tour, _)| tour.clone())
            .unwrap_or_default()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        "GA".to_string()
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
        assert_eq!(ga.population_costs.len(), 50);
        let (greedy_tour, greedy_cost) = ga.build_greedy_tour(0);
        assert_eq!(ga.population[0], greedy_tour);
        assert!((ga.population_costs[0] - greedy_cost).abs() < 1e-9);
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

        let expected_cost =
            tsp.weight(0, 1) + tsp.weight(1, 2) + tsp.weight(2, 3) + tsp.weight(3, 0);
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
        let mut ga = GeneticAlgorithm::with_options(tsp, 10, 2, 0.7, 0.01, 100);

        let parent1 = vec![0, 1, 2, 3, 4];
        let parent2 = vec![4, 3, 2, 1, 0];

        let child = GeneticAlgorithm::crossover_with_rng(&parent1, &parent2, &mut ga.rng);

        assert_eq!(child.len(), 5);
        assert!(child.iter().all(|&x| x < 5));
        assert_eq!(
            child.iter().collect::<std::collections::HashSet<_>>().len(),
            5
        );
    }

    #[test]
    fn uses_seeded_runs_deterministically() {
        let data = "
        NAME : example
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
        let mut lhs =
            GeneticAlgorithm::with_options_and_seed(tsp.clone(), 50, 5, 0.7, 0.01, 100, 7);
        let mut rhs = GeneticAlgorithm::with_options_and_seed(tsp, 50, 5, 0.7, 0.01, 100, 7);

        assert_eq!(lhs.seed(), 7);
        assert_eq!(lhs.solve(), rhs.solve());
    }

    #[test]
    fn evolve_returns_the_best_tour_in_the_new_population() {
        let data = "
        NAME : example
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
        let mut ga = GeneticAlgorithm::with_options_and_seed(tsp, 20, 4, 0.7, 0.01, 10, 17);

        let (best, best_cost) = ga.evolve();
        let population_best = ga
            .population_costs
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        assert!(
            (best_cost - population_best).abs() < 1e-9,
            "evolve must return the best member of the updated population"
        );
        assert!(!best.is_empty());
    }
}
