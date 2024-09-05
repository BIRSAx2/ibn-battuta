use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;

pub struct RedBlackACS {
    tsp: Tsp,
    pheromones_red: Vec<Vec<f64>>,
    pheromones_black: Vec<Vec<f64>>,
    heuristic: Vec<Vec<f64>>,
    best_tour_red: Vec<usize>,
    best_tour_black: Vec<usize>,
    best_cost_red: f64,
    best_cost_black: f64,
    alpha_red: f64,
    beta_red: f64,
    rho_red: f64,
    alpha_black: f64,
    beta_black: f64,
    rho_black: f64,
    tau0_red: f64,
    tau0_black: f64,
    q0_red: f64,
    q0_black: f64,
    num_ants: usize,
    max_iterations: usize,
}

impl RedBlackACS {
    pub fn with_options(tsp: Tsp, alpha_red: f64, beta_red: f64, rho_red: f64, tau0_red: f64, q0_red: f64,
                        alpha_black: f64, beta_black: f64, rho_black: f64, tau0_black: f64, q0_black: f64,
                        num_ants: usize, max_iterations: usize, c: f64) -> RedBlackACS {
        let dim = tsp.dim();
        let mut pheromones_red = vec![vec![0.0; dim]; dim];
        let mut pheromones_black = vec![vec![0.0; dim]; dim];
        let heuristic = vec![vec![0.0; dim]; dim];

        // Initialize pheromones based on edge costs for both red and black ants
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    pheromones_red[i][j] = c / tsp.weight(i, j);
                    pheromones_black[i][j] = c / tsp.weight(i, j);
                }
            }
        }

        let mut rb_acs = RedBlackACS {
            tsp,
            pheromones_red,
            pheromones_black,
            heuristic,
            best_tour_red: vec![],
            best_tour_black: vec![],
            best_cost_red: f64::INFINITY,
            best_cost_black: f64::INFINITY,
            alpha_red,
            beta_red,
            rho_red,
            alpha_black,
            beta_black,
            rho_black,
            tau0_red,
            tau0_black,
            q0_red,
            q0_black,
            num_ants,
            max_iterations,
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
            self.local_pheromone_update(&tour[i - 1..=i], is_red);
        }

        // Close the tour
        self.local_pheromone_update(&[tour[self.tsp.dim() - 1], tour[0]], is_red);

        tour
    }

    fn select_next_city(&self, partial_tour: &[usize], visited: &[bool], rng: &mut ThreadRng, is_red: bool) -> usize {
        let current_city = partial_tour[partial_tour.len() - 1];

        if is_red {
            if rng.gen::<f64>() < self.q0_red {
                self.select_best_city(current_city, visited, true)
            } else {
                self.select_probabilistic_city(current_city, visited, rng, true)
            }
        } else {
            if rng.gen::<f64>() < self.q0_black {
                self.select_best_city(current_city, visited, false)
            } else {
                self.select_probabilistic_city(current_city, visited, rng, false)
            }
        }
    }

    fn select_best_city(&self, current_city: usize, visited: &[bool], is_red: bool) -> usize {
        (0..self.tsp.dim())
            .filter(|&city| !visited[city])
            .max_by(|&a, &b| {
                let score_a = if is_red {
                    self.pheromones_red[current_city][a] * self.heuristic[current_city][a].powf(self.beta_red)
                } else {
                    self.pheromones_black[current_city][a] * self.heuristic[current_city][a].powf(self.beta_black)
                };
                let score_b = if is_red {
                    self.pheromones_red[current_city][b] * self.heuristic[current_city][b].powf(self.beta_red)
                } else {
                    self.pheromones_black[current_city][b] * self.heuristic[current_city][b].powf(self.beta_black)
                };
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap()
    }

    fn select_probabilistic_city(&self, current_city: usize, visited: &[bool], rng: &mut ThreadRng, is_red: bool) -> usize {
        let mut probabilities = vec![0.0; self.tsp.dim()];
        let mut total = 0.0;

        for (city, &visited) in visited.iter().enumerate() {
            if !visited {
                let probability = if is_red {
                    self.pheromones_red[current_city][city] * self.heuristic[current_city][city].powf(self.beta_red)
                } else {
                    self.pheromones_black[current_city][city] * self.heuristic[current_city][city].powf(self.beta_black)
                };
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

    fn local_pheromone_update(&mut self, edge: &[usize], is_red: bool) {
        let (i, j) = (edge[0], edge[1]);
        if is_red {
            self.pheromones_red[i][j] = (1.0 - self.rho_red) * self.pheromones_red[i][j] + self.rho_red * self.tau0_red;
            self.pheromones_red[j][i] = self.pheromones_red[i][j];
        } else {
            self.pheromones_black[i][j] = (1.0 - self.rho_black) * self.pheromones_black[i][j] + self.rho_black * self.tau0_black;
            self.pheromones_black[j][i] = self.pheromones_black[i][j];
        }
    }

    fn global_pheromone_update(&mut self) {
        let deposit_red = 1.0 / self.best_cost_red;
        let deposit_black = 1.0 / self.best_cost_black;

        for i in 0..self.best_tour_red.len() {
            let from = self.best_tour_red[i];
            let to = self.best_tour_red[(i + 1) % self.best_tour_red.len()];
            self.pheromones_red[from][to] = (1.0 - self.rho_red) * self.pheromones_red[from][to] + self.rho_red * deposit_red;
            self.pheromones_red[to][from] = self.pheromones_red[from][to];
        }

        for i in 0..self.best_tour_black.len() {
            let from = self.best_tour_black[i];
            let to = self.best_tour_black[(i + 1) % self.best_tour_black.len()];
            self.pheromones_black[from][to] = (1.0 - self.rho_black) * self.pheromones_black[from][to] + self.rho_black * deposit_black;
            self.pheromones_black[to][from] = self.pheromones_black[from][to];
        }
    }

    fn update_best_solution(&mut self, tour: &Vec<usize>, is_red: bool) {
        let cost = self.calculate_tour_cost(tour);
        if is_red {
            if cost < self.best_cost_red {
                self.best_tour_red = tour.clone();
                self.best_cost_red = cost;
            }
        } else {
            if cost < self.best_cost_black {
                self.best_tour_black = tour.clone();
                self.best_cost_black = cost;
            }
        }
    }

    fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
        let mut cost = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            cost += self.tsp.weight(from, to);
        }
        cost
    }
}

impl TspSolver for RedBlackACS {
    fn solve(&mut self) -> Solution {
        for _ in 0..self.max_iterations {
            for ant in 0..self.num_ants * 2 {
                let is_red = ant < self.num_ants;
                let solution = self.construct_solution(is_red);
                self.update_best_solution(&solution, is_red);
            }
            self.global_pheromone_update();
        }

        Solution {
            tour: if self.best_cost_red < self.best_cost_black {
                self.best_tour_red.clone()
            } else {
                self.best_tour_black.clone()
            },
            total: self.best_cost_red.min(self.best_cost_black),
        }
    }

    fn tour(&self) -> Vec<usize> {
        if self.best_cost_red < self.best_cost_black {
            self.best_tour_red.clone()
        } else {
            self.best_tour_black.clone()
        }
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
    fn format_name(&self) -> String {
        format!("RedBlackACS")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
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

        let mut nn = NearestNeighbor::new(tsp.clone());
        let base_tour = nn.solve().total;
        let n = tsp.dim();
        let tau0_red = 1.0 / (n as f64 * base_tour as f64);
        let tau0_black = 1.0 / (n as f64 * base_tour as f64);
        let mut solver = RedBlackACS::with_options(tsp, 1.0, 2.0, 0.1, tau0_red, 0.9, 1.2, 1.5, 0.2, tau0_black, 0.8, 20, 1000, 100.0);
        let solution = solver.solve();

        println!("{:?}", solution);
    }

    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();

        let mut nn = NearestNeighbor::new(tsp.clone());
        let base_tour = nn.solve().total;
        let n = tsp.dim();
        let tau0_red = 1.0 / (n as f64 * base_tour as f64);
        let tau0_black = 1.0 / (n as f64 * base_tour as f64);
        let mut solver = RedBlackACS::with_options(tsp, 1.0, 2.0, 0.1, tau0_red, 0.9, 1.2, 1.5, 0.2, tau0_black, 0.8, 20, 1000, 100.0);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
    #[test]
    fn test_p43() {
        let path = "data/tsplib/gr666.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        test_instance(tsp);
    }
}
