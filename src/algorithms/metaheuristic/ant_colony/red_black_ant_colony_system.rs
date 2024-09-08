use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use crate::NearestNeighbor;
use rand::prelude::*;
use std::{cmp::Ordering, f64, mem};

#[allow(dead_code)]
pub struct RedBlackACS {
    tsp: Tsp,
    pheromones_red: Vec<Vec<f64>>,
    pheromones_black: Vec<Vec<f64>>,
    heuristic: Vec<Vec<f64>>,
    candidate_lists: Vec<Vec<usize>>,
    best_tour_red: Vec<usize>,
    best_tour_black: Vec<usize>,
    best_cost_red: f64,
    best_cost_black: f64,
    alpha: f64, // Used for pheromone influence
    beta: f64,
    rho_red: f64,
    rho_black: f64,
    tau0: f64,
    q0: f64,
    num_ants: usize,
    max_iterations: usize,
    candidate_list_size: usize,
}

impl RedBlackACS {
    pub fn new(tsp: Tsp, alpha: f64, beta: f64, rho_red: f64, rho_black: f64, q0: f64,
               num_ants: usize, max_iterations: usize, candidate_list_size: usize) -> RedBlackACS {
        let dim = tsp.dim();
        let mut pheromones_red = vec![vec![0.0; dim]; dim];
        let mut pheromones_black = vec![vec![0.0; dim]; dim];
        let heuristic = vec![vec![0.0; dim]; dim];

        // Calculate tau0 based on nearest neighbor heuristic
        let nn_tour_length = NearestNeighbor::new(tsp.clone()).solve().total;
        let tau0 = 1.0 / (dim as f64 * nn_tour_length);

        // Initialize pheromones inversely proportional to edge weights
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    pheromones_red[i][j] = tau0 / tsp.weight(i, j);
                    pheromones_black[i][j] = tau0 / tsp.weight(i, j);
                }
            }
        }

        let mut rb_acs = RedBlackACS {
            tsp,
            pheromones_red,
            pheromones_black,
            heuristic,
            candidate_lists: vec![],
            best_tour_red: vec![],
            best_tour_black: vec![],
            best_cost_red: f64::INFINITY,
            best_cost_black: f64::INFINITY,
            alpha,
            beta,
            rho_red,
            rho_black,
            tau0,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
        };

        rb_acs.initialize_heuristic();
        rb_acs.initialize_candidate_lists();
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

    fn initialize_candidate_lists(&mut self) {
        let n = self.tsp.dim();
        self.candidate_lists = vec![vec![]; n];

        for i in 0..n {
            let mut candidates: Vec<(usize, f64)> = (0..n)
                .filter(|&j| i != j)
                .map(|j| (j, self.tsp.weight(i, j)))
                .collect();

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            self.candidate_lists[i] = candidates.into_iter()
                .take(self.candidate_list_size)
                .map(|(j, _)| j)
                .collect();
        }
    }

    fn construct_solution(&mut self, is_red: bool) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut tour = vec![0; self.tsp.dim()];
        let mut visited = vec![false; self.tsp.dim()];

        // Randomly select starting city
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
        let pheromones = if is_red { &self.pheromones_red } else { &self.pheromones_black };

        if rng.gen::<f64>() < self.q0 {
            // Use alpha in the best city selection
            self.select_best_city(current_city, visited, pheromones)
        } else {
            self.select_probabilistic_city(current_city, visited, rng, pheromones)
        }
    }

    fn select_best_city(&self, current_city: usize, visited: &[bool], pheromones: &[Vec<f64>]) -> usize {
        self.candidate_lists[current_city]
            .iter()
            .filter(|&&city| !visited[city])
            .max_by(|&&a, &&b| {
                let score_a = pheromones[current_city][a].powf(self.alpha) * self.heuristic[current_city][a].powf(self.beta);
                let score_b = pheromones[current_city][b].powf(self.alpha) * self.heuristic[current_city][b].powf(self.beta);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .cloned()
            .unwrap_or_else(|| {
                (0..self.tsp.dim())
                    .filter(|&city| !visited[city])
                    .max_by(|&a, &b| {
                        let score_a = pheromones[current_city][a].powf(self.alpha) * self.heuristic[current_city][a].powf(self.beta);
                        let score_b = pheromones[current_city][b].powf(self.alpha) * self.heuristic[current_city][b].powf(self.beta);
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .unwrap()
            })
    }

    fn select_probabilistic_city(&self, current_city: usize, visited: &[bool], rng: &mut ThreadRng, pheromones: &[Vec<f64>]) -> usize {
        let mut probabilities = vec![0.0; self.tsp.dim()];
        let mut total = 0.0;

        for &city in &self.candidate_lists[current_city] {
            if !visited[city] {
                let probability = pheromones[current_city][city].powf(self.alpha) * self.heuristic[current_city][city].powf(self.beta);
                probabilities[city] = probability;
                total += probability;
            }
        }

        if total == 0.0 {
            for city in 0..self.tsp.dim() {
                if !visited[city] {
                    let probability = pheromones[current_city][city].powf(self.alpha) * self.heuristic[current_city][city].powf(self.beta);
                    probabilities[city] = probability;
                    total += probability;
                }
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
        let (pheromones, rho) = if is_red {
            (&mut self.pheromones_red, self.rho_red)
        } else {
            (&mut self.pheromones_black, self.rho_black)
        };
        pheromones[i][j] = (1.0 - rho) * pheromones[i][j] + rho * self.tau0;
        pheromones[j][i] = pheromones[i][j];
    }

    fn global_pheromone_update(&mut self) {
        let deposit_red = 1.0 / self.best_cost_red;
        let deposit_black = 1.0 / self.best_cost_black;

        // Update pheromones for the best two ants in each group
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

    fn update_best_solution(&mut self, tour: &mut Vec<usize>, is_red: bool) {
        let cost = self.calculate_tour_cost(tour);
        if is_red {
            if cost < self.best_cost_red {
                mem::swap(&mut self.best_tour_red, tour);
                self.best_cost_red = cost;
            }
        } else {
            if cost < self.best_cost_black {
                mem::swap(&mut self.best_tour_black, tour);
                self.best_cost_black = cost;
            }
        }
    }

    fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
        tour.windows(2).map(|w| self.tsp.weight(w[0], w[1])).sum::<f64>()
            + self.tsp.weight(*tour.last().unwrap(), tour[0])
    }
}

impl TspSolver for RedBlackACS {
    fn solve(&mut self) -> Solution {
        for _ in 0..self.max_iterations {
            for ant in 0..self.num_ants * 2 {
                let is_red = ant < self.num_ants;
                let mut solution = self.construct_solution(is_red);
                self.update_best_solution(&mut solution, is_red);
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

    #[inline(always)]
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

        let mut solver = RedBlackACS::new(tsp, 1.0, 2.0, 0.1, 0.2, 0.9, 20, 1000, 3);
        let solution = solver.solve();

        println!("{:?}", solution);
    }

    #[test]
    fn test_gr666() {
        let path = "data/tsplib/a280.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        let mut solver = RedBlackACS::new(tsp, 1.0, 2.0, 0.1, 0.2, 0.9, 10, 1000, 15);
        let solution = solver.solve();
        println!("{:?}", solution);
    }
}
