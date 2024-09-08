use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::cmp::Ordering;
use std::f64;

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

    fn calculate_fitness(&self, tour: &[usize]) -> f64 {
        1.0 / self.calculate_tour_cost(tour)
    }

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

    fn mutate(&self, tour: &mut Vec<usize>) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.mutation_rate {
            let i = rng.gen_range(0..tour.len());
            let j = rng.gen_range(0..tour.len());
            tour.swap(i, j);
        }
    }

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
            total: best_cost,
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

        let mut solver = GeneticAlgorithm::with_options(tsp, 100, 5, 0.7, 0.01, 500);
        let solution = solver.solve();

        println!("Example solution: {:?}", solution);
        assert_eq!(solution.tour.len(), dim);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        run_on_instance(tsp);
    }

    #[test]
    fn test_berlin52() {
        let path = "data/tsplib/st70.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        run_on_instance(tsp);
    }

    fn run_on_instance(tsp: Tsp) {
        let m = tsp.dim();

        let mut solver = GeneticAlgorithm::with_options(tsp, 70, (70.0 * 0.05) as usize, 0.8, 0.02, 1000);

        let solution = solver.solve();

        println!("Solution: {:?}", solution);
        assert_eq!(solution.tour.len(), m);
    }
}