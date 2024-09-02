use std::f64;
use tspf::{Tsp};
use crate::algorithms::{Solution, SolverOptions, TspSolver};
use rand::prelude::*;
use crate::algorithms::metaheuristic::simulated_annealing::SimulatedAnnealing;

pub struct GeneticAlgorithm<'a> {
    tsp: &'a Tsp,
    population: Vec<Vec<usize>>,
    best_tour: Vec<usize>,
    best_cost: f64,
    options: SolverOptions,
}

impl<'a> GeneticAlgorithm<'a> {
    pub fn new(tsp: &'a Tsp, options: SolverOptions) -> GeneticAlgorithm<'a> {
        GeneticAlgorithm {
            tsp,
            population: vec![],
            best_tour: vec![],
            best_cost: f64::INFINITY,
            options,
        }
    }

    fn initialize_population(&mut self) {
        let mut rng = rand::thread_rng();
        for _ in 0..self.options.population_size {
            let mut tour: Vec<usize> = (0..self.tsp.dim()).collect();
            tour.shuffle(&mut rng);
            self.population.push(tour);
        }
    }

    fn calculate_fitness(&self, tour: &Vec<usize>) -> f64 {
        let cost = self.calculate_tour_cost(tour);
        1.0 / cost
    }

    fn select_parent(&self) -> &Vec<usize> {
        let mut rng = rand::thread_rng();
        let tournament_size = self.options.tournament_size;
        let mut best = None;
        let mut best_fitness = 0.0;

        for _ in 0..tournament_size {
            let candidate = self.population.choose(&mut rng).unwrap();
            let fitness = self.calculate_fitness(candidate);
            if fitness > best_fitness {
                best = Some(candidate);
                best_fitness = fitness;
            }
        }

        best.unwrap()
    }

    fn crossover(&self, parent1: &Vec<usize>, parent2: &Vec<usize>) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut child = vec![usize::MAX; self.tsp.dim()];
        let start = rng.gen_range(0..self.tsp.dim());
        let end = rng.gen_range(start..self.tsp.dim());

        for i in start..=end {
            child[i] = parent1[i];
        }

        let mut j = (end + 1) % self.tsp.dim();
        for i in 0..self.tsp.dim() {
            if !child.contains(&parent2[i]) {
                child[j] = parent2[i];
                j = (j + 1) % self.tsp.dim();
            }
        }

        child
    }

    fn mutate(&self, tour: &mut Vec<usize>) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.options.mutation_rate {
            let i = rng.gen_range(0..tour.len());
            let j = rng.gen_range(0..tour.len());
            tour.swap(i, j);
        }
    }

    fn evolve_population(&mut self) {
        let mut new_population = Vec::with_capacity(self.population.len());

        for _ in 0..self.population.len() {
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
    fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
        let mut total_cost = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total_cost += self.cost(from, to);
        }
        total_cost
    }
}

impl TspSolver for GeneticAlgorithm<'_> {
    fn solve(&mut self, options: &SolverOptions) -> Solution {
        self.options = options.clone();
        self.initialize_population();

        for _ in 0..self.options.max_generations {
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
    use tspf::TspBuilder;
    use crate::algorithms::{SolverOptions, TspSolver};
    use crate::algorithms::heuristic::local_search::two_opt::TwoOpt;
    use super::*;


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

        let mut options = SolverOptions::default();
        options.verbose = true;
        let mut solver = GeneticAlgorithm::new(&tsp, options);
        let solution = solver.solve(&SolverOptions::default());

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let options = SolverOptions::default();
        let mut solver = GeneticAlgorithm::new(&tsp, options);
        let solution = solver.solve(&SolverOptions::default());
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}