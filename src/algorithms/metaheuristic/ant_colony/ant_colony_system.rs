use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use crate::NearestNeighbor;
use rand::prelude::*;
use std::cmp::Ordering;
use std::f64;

pub struct AntColonySystem {
    tsp: Tsp,
    pheromones: Vec<Vec<f64>>,
    heuristic: Vec<Vec<f64>>,
    best_tour: Vec<usize>,
    best_cost: f64,
    candidate_lists: Vec<Vec<usize>>,

    // params
    alpha: f64,
    beta: f64,
    rho: f64,
    tau0: f64,
    q0: f64,
    num_ants: usize,
    max_iterations: usize,
    candidate_list_size: usize,
}

impl AntColonySystem {
    pub fn with_options(tsp: Tsp, alpha: f64, beta: f64, rho: f64, q0: f64, num_ants: usize, max_iterations: usize, candidate_list_size: usize) -> AntColonySystem {
        let mut nn = NearestNeighbor::new(tsp.clone());
        let base_tour = nn.solve().total;
        let n = tsp.dim();
        let tau0 = 1.0 / (n as f64 * base_tour);

        let pheromones = vec![vec![tau0; n]; n];
        let heuristic = vec![vec![0.0; n]; n];

        let mut acs = AntColonySystem {
            tsp,
            pheromones,
            heuristic,
            best_tour: vec![],
            best_cost: f64::INFINITY,
            candidate_lists: vec![],

            alpha,
            beta,
            rho,
            tau0,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
        };

        acs.initialize_heuristic();
        acs.initialize_candidate_lists();
        acs
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

    fn construct_solution(&mut self) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut tour = vec![0; self.tsp.dim()];
        let mut visited = vec![false; self.tsp.dim()];

        tour[0] = rng.gen_range(0..self.tsp.dim());
        visited[tour[0]] = true;

        for i in 1..self.tsp.dim() {
            tour[i] = self.select_next_city(&tour[0..i], &visited, &mut rng);
            visited[tour[i]] = true;
            self.local_pheromone_update(&tour[i - 1..=i]);
        }

        // Close the tour
        self.local_pheromone_update(&[tour[self.tsp.dim() - 1], tour[0]]);

        tour
    }

    fn select_next_city(&self, partial_tour: &[usize], visited: &[bool], rng: &mut ThreadRng) -> usize {
        let current_city = partial_tour[partial_tour.len() - 1];

        if rng.gen::<f64>() < self.q0 {
            // Exploitation (choose best)
            self.select_best_city(current_city, visited)
        } else {
            // Exploration (probabilistic choice)
            self.select_probabilistic_city(current_city, visited, rng)
        }
    }

    fn select_best_city(&self, current_city: usize, visited: &[bool]) -> usize {
        self.candidate_lists[current_city]
            .iter()
            .filter(|&&city| !visited[city])
            .max_by(|&&a, &&b| {
                let score_a = self.pheromones[current_city][a] * self.heuristic[current_city][a].powf(self.beta);
                let score_b = self.pheromones[current_city][b] * self.heuristic[current_city][b].powf(self.beta);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .cloned()
            .unwrap_or_else(|| {
                // If all candidates are visited, choose the best among all unvisited cities
                (0..self.tsp.dim())
                    .filter(|&city| !visited[city])
                    .max_by(|&a, &b| {
                        let score_a = self.pheromones[current_city][a] * self.heuristic[current_city][a].powf(self.beta);
                        let score_b = self.pheromones[current_city][b] * self.heuristic[current_city][b].powf(self.beta);
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .unwrap()
            })
    }

    fn select_probabilistic_city(&self, current_city: usize, visited: &[bool], rng: &mut ThreadRng) -> usize {
        let mut probabilities = vec![0.0; self.tsp.dim()];
        let mut total = 0.0;

        for &city in &self.candidate_lists[current_city] {
            if !visited[city] {
                let probability = self.pheromones[current_city][city] * self.heuristic[current_city][city].powf(self.beta);
                probabilities[city] = probability;
                total += probability;
            }
        }

        if total == 0.0 {
            // If all candidates are visited, consider all unvisited cities
            for city in 0..self.tsp.dim() {
                if !visited[city] {
                    let probability = self.pheromones[current_city][city] * self.heuristic[current_city][city].powf(self.beta);
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

        // Fallback in case of floating-point precision issues
        visited.iter().position(|&v| !v).unwrap()
    }

    fn local_pheromone_update(&mut self, edge: &[usize]) {
        let (i, j) = (edge[0], edge[1]);
        self.pheromones[i][j] = (1.0 - self.rho) * self.pheromones[i][j] + self.rho * self.tau0;
        self.pheromones[j][i] = self.pheromones[i][j];
    }

    fn global_pheromone_update(&mut self) {
        let deposit = 1.0 / self.best_cost;

        // Evaporation on all edges
        for i in 0..self.tsp.dim() {
            for j in 0..self.tsp.dim() {
                self.pheromones[i][j] *= 1.0 - self.alpha;
            }
        }

        // Pheromone update only for the best tour
        for i in 0..self.best_tour.len() {
            let from = self.best_tour[i];
            let to = self.best_tour[(i + 1) % self.best_tour.len()];

            self.pheromones[from][to] += self.alpha * deposit;
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

impl TspSolver for AntColonySystem {
    fn solve(&mut self) -> Solution {
        for _ in 0..self.max_iterations {
            for _ in 0..self.num_ants {
                let solution = self.construct_solution();
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

    fn format_name(&self) -> String {
        format!("AntColonySystem")
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::metaheuristic::ant_colony::ant_colony_system::AntColonySystem;
    use crate::algorithms::TspSolver;
    use crate::{Tsp, TspBuilder};

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
        let n = tsp.dim();
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, n, 1000, 3);

        let solution = solver.solve();

        println!("{:?}", solution);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let n = tsp.dim();
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, n, 1000, 5);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), n);
    }

    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();

        // let n = tsp.dim();
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 10, 1000, 15);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }

    #[test]
    fn test_gr666() {
        let path = "data/tsplib/st70.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        test_instance(tsp);
    }

    #[test]
    fn test_p43() {
        let path = "data/tsplib/berlin52.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        test_instance(tsp);
    }
}