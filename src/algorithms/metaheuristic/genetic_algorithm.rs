use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;
pub struct GeneticAlgorithm {
    tsp: Tsp,
    population: Vec<Vec<usize>>,
    best_tour: Vec<usize>,
    best_cost: f64,
    population_size: usize,
    tournament_size: usize,
    mutation_rate: f64,
    max_generations: usize,
}

impl GeneticAlgorithm {
    pub fn with_options(
        tsp: Tsp,
        population_size: usize,
        tournament_size: usize,
        mutation_rate: f64,
        max_generations: usize,
    ) -> GeneticAlgorithm {
        let population_size = population_size.max(2);  // Ensure at least 2 individuals
        let tournament_size = tournament_size.min(population_size);  // Ensure tournament size doesn't exceed population size

        let mut ga = GeneticAlgorithm {
            tsp,
            population: Vec::with_capacity(population_size),
            best_tour: vec![],
            best_cost: f64::INFINITY,
            population_size,
            tournament_size,
            mutation_rate,
            max_generations,
        };
        ga.initialize_population();
        ga
    }

    fn initialize_population(&mut self) {
        let mut rng = rand::thread_rng();
        for _ in 0..self.population_size {
            let mut tour: Vec<usize> = (0..self.tsp.dim()).collect();
            tour.shuffle(&mut rng);
            self.population.push(tour);
        }
    }

    fn calculate_fitness(&self, tour: &[usize]) -> f64 {
        let cost = self.calculate_tour_cost(tour);
        1.0 / cost
    }

    fn select_parent(&self) -> &[usize] {
        let mut rng = rand::thread_rng();
        self.population
            .choose_multiple(&mut rng, self.tournament_size)
            .max_by(|&a, &b| {
                self.calculate_fitness(a)
                    .partial_cmp(&self.calculate_fitness(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Population is empty")
    }

    fn crossover(&self, parent1: &[usize], parent2: &[usize]) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut child = vec![usize::MAX; self.tsp.dim()];
        let start = rng.gen_range(0..self.tsp.dim());
        let end = rng.gen_range(start..self.tsp.dim());

        child[start..=end].copy_from_slice(&parent1[start..=end]);

        let mut j = (end + 1) % self.tsp.dim();
        for &city in parent2.iter().chain(parent2.iter()) {
            if !child[start..=end].contains(&city) {
                child[j] = city;
                j = (j + 1) % self.tsp.dim();
                if j == start {
                    break;
                }
            }
        }

        child
    }

    fn mutate(&self, tour: &mut [usize]) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.mutation_rate {
            let i = rng.gen_range(0..tour.len());
            let j = rng.gen_range(0..tour.len());
            tour.swap(i, j);
        }
    }

    fn evolve_population(&mut self) {
        let mut new_population = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            let parent1 = self.select_parent();
            let parent2 = self.select_parent();
            let mut child = self.crossover(parent1, parent2);
            self.mutate(&mut child);
            new_population.push(child);
        }

        self.population = new_population;
    }

    fn update_best_solution(&mut self) {
        for tour in &self.population {
            let cost = self.calculate_tour_cost(tour);
            if cost < self.best_cost {
                self.best_tour = tour.clone();
                self.best_cost = cost;
            }
        }
    }

    fn calculate_tour_cost(&self, tour: &[usize]) -> f64 {
        tour.windows(2)
            .map(|w| self.cost(w[0], w[1]))
            .sum::<f64>()
            + self.cost(*tour.last().unwrap(), tour[0])
    }
}

impl TspSolver for GeneticAlgorithm {
    fn solve(&mut self) -> Solution {
        for _ in 0..self.max_generations {
            self.evolve_population();
            self.update_best_solution();
        }

        Solution {
            tour: self.best_tour.clone(),
            total: self.best_cost,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TspBuilder;

    #[test]
    fn test_example() {
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
        let dim = tsp.dim();

        let mut solver = GeneticAlgorithm::with_options(tsp, 100, 5, 0.01, 1000);
        let solution = solver.solve();

        println!("Example solution: {:?}", solution);
        assert_eq!(solution.tour.len(), dim);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        let m = tsp.dim();

        let mut solver = GeneticAlgorithm::with_options(tsp, 100, 5, 0.01, 1000);
        let solution = solver.solve();

        println!("GR17 solution: {:?}", solution);
        assert_eq!(solution.tour.len(), m);
    }
}