use crate::algorithms::utils::{MetaheuristicAlgorithmConfig, SolverConfig};
use crate::algorithms::{Solution, TspSolver};
use rand::prelude::*;
use std::f64;
use tspf::Tsp;

pub struct SimulatedAnnealing<'a> {
    tsp: &'a Tsp,
    tour: Vec<usize>,
    cost: f64,
    initial_temperature: f64,
    cooling_rate: f64,
    min_temperature: f64,
    max_iterations: usize,
}

impl<'a> SimulatedAnnealing<'a> {
    pub fn new(tsp: &'a Tsp) -> SimulatedAnnealing<'a> {
        SimulatedAnnealing {
            tsp,
            tour: vec![],
            cost: 0.0,
            initial_temperature: 1000.0,
            cooling_rate: 0.003,
            min_temperature: 0.0001,
            max_iterations: 1000,
        }
    }

    fn initial_solution(&mut self) {
        let mut rng = rand::thread_rng();
        self.tour = (0..self.tsp.dim()).collect();
        self.tour.shuffle(&mut rng);
        self.cost = self.calculate_tour_cost(&self.tour);
    }

    // fn calculate_tour_cost(&self) -> f64 {
    //     let mut total_cost = 0.0;
    //     for i in 0..self.tour.len() {
    //         let from = self.tour[i];
    //         let to = self.tour[(i + 1) % self.tour.len()];
    //         total_cost += self.cost(from, to);
    //     }
    //     total_cost
    // }

    fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
        let mut total_cost = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total_cost += self.cost(from, to);
        }
        total_cost
    }

    fn generate_neighbor(&self) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut new_tour = self.tour.clone();
        let i = rng.gen_range(0..new_tour.len());
        let j = rng.gen_range(0..new_tour.len());
        new_tour.swap(i, j);
        new_tour
    }

    fn acceptance_probability(old_cost: f64, new_cost: f64, temperature: f64) -> f64 {
        if new_cost < old_cost {
            1.0
        } else {
            f64::exp((old_cost - new_cost) / temperature)
        }
    }
}

impl TspSolver for SimulatedAnnealing<'_> {
    fn solve(&mut self, options: &SolverConfig) -> Solution {
        (self.initial_temperature, self.cooling_rate, self.min_temperature, self.max_iterations) = match options {
            SolverConfig::MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig::SimulatedAnnealing {
                                                     initial_temperature,
                                                     cooling_rate,
                                                     min_temperature,
                                                     max_iterations,
                                                 }) => (*initial_temperature, *cooling_rate, *min_temperature, *max_iterations),
            _ => (1000.0, 0.003, 0.0001, 1000),
        };
        let mut rng = rand::thread_rng();
        self.initial_solution();

        let mut best_tour = self.tour.clone();
        let mut best_cost = self.cost;

        let mut temperature = self.initial_temperature;
        let cooling_rate = self.cooling_rate;

        for _ in 0..self.max_iterations {
            let new_tour = self.generate_neighbor();
            let new_cost = self.calculate_tour_cost(&new_tour);

            if SimulatedAnnealing::acceptance_probability(self.cost, new_cost, temperature) > rng.gen() {
                self.tour = new_tour;
                self.cost = new_cost;
            }

            if self.cost < best_cost {
                best_tour = self.tour.clone();
                best_cost = self.cost;
            }

            temperature *= 1.0 - cooling_rate;

            if temperature < self.min_temperature {
                break;
            }
        }

        self.tour = best_tour;
        self.cost = best_cost;

        Solution {
            tour: self.tour.clone(),
            total: self.cost,
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
    use crate::algorithms::{SolverConfig, TspSolver};
    use tspf::TspBuilder;


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

        let options = SolverConfig::new_simulated_annealing(1000.0, 0.003, 0.0001, 1000);
        let mut solver = SimulatedAnnealing::new(&tsp);
        let solution = solver.solve(&options);

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let options = SolverConfig::new_simulated_annealing(1000.0, 0.003, 0.0001, 1000);
        let mut solver = SimulatedAnnealing::new(&tsp);
        let solution = solver.solve(&options);
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}