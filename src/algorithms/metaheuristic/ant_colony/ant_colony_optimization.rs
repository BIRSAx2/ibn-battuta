use crate::algorithms::{Solution, TspSolver};
use rand::prelude::*;
use std::f64;
use tspf::Tsp;

pub struct AntSystem<'a> {
    tsp: &'a Tsp,
    pheromones: Vec<Vec<f64>>,
    best_tour: Vec<usize>,
    best_cost: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    num_ants: usize,
    max_iterations: usize,
}

impl<'a> AntSystem<'a> {
    pub fn new(tsp: &'a Tsp, alpha: f64, beta: f64, rho: f64, num_ants: usize, max_iterations: usize) -> AntSystem<'a> {
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

    fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
        let mut total_cost = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total_cost += self.cost(from, to);
        }
        total_cost
    }

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

impl TspSolver for AntSystem<'_> {
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

        let mut solver = AntSystem::new(&tsp, 1.0, 2.0, 0.5, 10, 100);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), tsp.dim());
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let mut solver = AntSystem::new(&tsp, 1.0, 2.0, 0.5, 20, 1000);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), tsp.dim());
    }
}