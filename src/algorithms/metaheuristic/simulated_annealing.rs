use crate::algorithms::NearestNeighbor;
use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;

pub struct SimulatedAnnealing {
    tsp: Tsp,
    tour: Vec<usize>,
    cost: f64,
    initial_temperature: f64,
    cooling_rate: f64,
    min_temperature: f64,
    max_temperature_changes: usize,
    cycles_per_temperature: usize,
}

impl SimulatedAnnealing {
    pub fn with_options(
        tsp: Tsp,
        initial_temperature: f64,
        cooling_rate: f64,
        min_temperature: f64,
        max_temperature_changes: usize,
        cycles_per_temperature: usize,
    ) -> SimulatedAnnealing {
        SimulatedAnnealing {
            tsp,
            tour: vec![],
            cost: 0.0,
            initial_temperature,
            cooling_rate,
            min_temperature,
            max_temperature_changes,
            cycles_per_temperature,
        }
    }

    fn initial_solution(&mut self) {
        let solution = NearestNeighbor::new(self.tsp.clone()).solve();
        self.tour = solution.tour;
        self.cost = solution.total;
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

    fn generate_neighbor(&self) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut new_tour = self.tour.clone();
        let i = rng.gen_range(0..new_tour.len());
        let mut j = rng.gen_range(0..new_tour.len());
        while j == i {
            j = rng.gen_range(0..new_tour.len());
        }
        new_tour.swap(i, j);
        new_tour
    }

    fn acceptance_probability(old_cost: f64, new_cost: f64, temperature: f64) -> f64 {
        f64::exp(-(new_cost - old_cost) / temperature)
    }
}

impl TspSolver for SimulatedAnnealing {
    fn solve(&mut self) -> Solution {
        let mut rng = rand::thread_rng();
        self.initial_solution();

        let mut best_tour = self.tour.clone();
        let mut best_cost = self.cost;

        let mut temperature = self.initial_temperature;

        for _ in 0..self.max_temperature_changes {
            for _ in 0..self.cycles_per_temperature {
                let new_tour = self.generate_neighbor();
                let new_cost = self.calculate_tour_cost(&new_tour);

                if SimulatedAnnealing::acceptance_probability(self.cost, new_cost, temperature) > rng.gen() {
                    self.tour = new_tour;
                    self.cost = new_cost;

                    if self.cost < best_cost {
                        best_tour = self.tour.clone();
                        best_cost = self.cost;
                    }
                }
            }

            temperature *= self.cooling_rate;

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

    fn format_name(&self) -> String {
        format!("SimulatedAnnealing")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::TspSolver;
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
        let mut solver = SimulatedAnnealing::with_options(tsp.clone(), 100.0, 0.98, 1e-8, tsp.dim() * 100, 10);
        let solution = solver.solve();

        println!("{:?}", solution);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/bier127.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let sol = test_instance(tsp);
        let best_known = 118282.0;
        let gap = (sol.total - best_known) / best_known;
        println!("Gap: {:.2}%", gap * 100.0);
    }

    #[test]
    fn test_gr120() {
        let path = "data/tsplib/berlin52.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        test_instance(tsp);
    }

    fn test_instance(tsp: Tsp) -> Solution {
        let size = tsp.dim();
        let mut solver = SimulatedAnnealing::with_options(tsp, 1000.0, 0.999, 0.0001, 1000, 100);
        let solution = solver.solve();
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
        solution
    }
}