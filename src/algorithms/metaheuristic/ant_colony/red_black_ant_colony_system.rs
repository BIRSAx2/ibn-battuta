use std::f64;
use tspf::{Tsp};
use crate::algorithms::{Solution, SolverOptions, TspSolver};
use rand::prelude::*;
use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
use crate::algorithms::metaheuristic::ant_colony::ant_colony_system::AntColonySystem;

pub struct RedBlackACS<'a> {
    tsp: &'a Tsp,
    pheromones: Vec<Vec<f64>>,
    heuristic: Vec<Vec<f64>>,
    best_tour: Vec<usize>,
    best_cost: f64,
    options: SolverOptions,
}

impl<'a> RedBlackACS<'a> {
    pub fn new(tsp: &'a Tsp, options: SolverOptions) -> RedBlackACS<'a> {
        let dim = tsp.dim();
        let mut nn = NearestNeighbor::new(&tsp);
        let nn_cost = nn.solve(&SolverOptions::default()).total;
        let initial_pheromone = 1.0 / (dim as f64 * nn_cost);
        let pheromones = vec![vec![initial_pheromone; dim]; dim];
        let heuristic = vec![vec![0.0; dim]; dim];

        let mut rb_acs = RedBlackACS {
            tsp,
            pheromones,
            heuristic,
            best_tour: vec![],
            best_cost: f64::INFINITY,
            options,
        };

        rb_acs.initialize_heuristic();
        rb_acs
    }

    fn initialize_heuristic(&mut self) {
        for i in 0..self.tsp.dim() {
            for j in 0..self.tsp.dim() {
                if i != j {
                    self.heuristic[i][j] = 1.0 / self.tsp.weight(i, j);
                }
            }
        }
    }

    fn construct_solution(&mut self, is_red: bool) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut tour = vec![0; self.tsp.dim()];
        let mut visited = vec![false; self.tsp.dim()];

        tour[0] = rng.gen_range(0..self.tsp.dim());
        visited[tour[0]] = true;

        for i in 1..self.tsp.dim() {
            tour[i] = self.select_next_city(&tour[0..i], &visited, &mut rng, is_red);
            visited[tour[i]] = true;
            self.local_pheromone_update(&tour[i - 1..=i]);
        }

        // Close the tour
        self.local_pheromone_update(&[tour[self.tsp.dim() - 1], tour[0]]);

        tour
    }

    fn select_next_city(&self, partial_tour: &[usize], visited: &[bool], rng: &mut ThreadRng, is_red: bool) -> usize {
        let current_city = partial_tour[partial_tour.len() - 1];

        if is_red {
            // Red ants always choose the best option (exploitation)
            self.select_best_city(current_city, visited)
        } else {
            // Black ants use the ACS rule (balancing exploitation and exploration)
            if rng.gen::<f64>() < self.options.q0 {
                self.select_best_city(current_city, visited)
            } else {
                self.select_probabilistic_city(current_city, visited, rng)
            }
        }
    }

    fn select_best_city(&self, current_city: usize, visited: &[bool]) -> usize {
        (0..self.tsp.dim())
            .filter(|&city| !visited[city])
            .max_by(|&a, &b| {
                let score_a = self.pheromones[current_city][a] * self.heuristic[current_city][a].powf(self.options.beta);
                let score_b = self.pheromones[current_city][b] * self.heuristic[current_city][b].powf(self.options.beta);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap()
    }

    fn select_probabilistic_city(&self, current_city: usize, visited: &[bool], rng: &mut ThreadRng) -> usize {
        let mut probabilities = vec![0.0; self.tsp.dim()];
        let mut total = 0.0;

        for (city, &visited) in visited.iter().enumerate() {
            if !visited {
                let probability = self.pheromones[current_city][city] * self.heuristic[current_city][city].powf(self.options.beta);
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

        // Fallback in case of floating-point precision issues
        visited.iter().position(|&v| !v).unwrap()
    }

    fn local_pheromone_update(&mut self, edge: &[usize]) {
        let (i, j) = (edge[0], edge[1]);
        self.pheromones[i][j] = (1.0 - self.options.rho) * self.pheromones[i][j] + self.options.rho * self.options.tau0;
        self.pheromones[j][i] = self.pheromones[i][j];
    }

    fn global_pheromone_update(&mut self) {
        let deposit = 1.0 / self.best_cost;

        for i in 0..self.best_tour.len() {
            let from = self.best_tour[i];
            let to = self.best_tour[(i + 1) % self.best_tour.len()];

            self.pheromones[from][to] = (1.0 - self.options.alpha) * self.pheromones[from][to] + self.options.alpha * deposit;
            self.pheromones[to][from] = self.pheromones[from][to];
        }
    }

    fn update_best_solution(&mut self, tour: &Vec<usize>) {
        let cost = self.calculate_tour_cost(tour);
        if cost < self.best_cost {
            self.best_tour = tour.clone();
            self.best_cost = cost;
        }
    }
}

impl TspSolver for RedBlackACS<'_> {
    fn solve(&mut self, options: &SolverOptions) -> Solution {
        self.options = options.clone();

        for _ in 0..self.options.max_iterations {
            for ant in 0..self.options.num_ants {
                let is_red = ant < self.options.num_red_ants;
                let solution = self.construct_solution(is_red);
                self.update_best_solution(&solution);
            }
            self.global_pheromone_update();
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
        let mut solver = RedBlackACS::new(&tsp, options);
        let solution = solver.solve(&SolverOptions::default());

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let options = SolverOptions::default();
        let mut solver = RedBlackACS::new(&tsp, options);
        let solution = solver.solve(&SolverOptions::default());
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}