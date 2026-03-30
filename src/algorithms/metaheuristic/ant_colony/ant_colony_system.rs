use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use crate::NearestNeighbor;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cmp::Ordering;
use std::f64;
use std::mem;

// Represents the Ant Colony System (ACS) algorithm for solving the Traveling Salesman Problem (TSP).
///
/// This implementation is based on the paper by Dorigo et al.
///
/// # Example
///
/// ```
/// use ibn_battuta::experimental::AntColonySystem;
/// use ibn_battuta::{TspBuilder, TspSolver};
///
/// let tsp = TspBuilder::parse_str("
///     NAME : example
///     TYPE : TSP
///     DIMENSION : 5
///     EDGE_WEIGHT_TYPE: EUC_2D
///     NODE_COORD_SECTION
///       1 1.2 3.4
///       2 5.6 7.8
///       3 3.4 5.6
///       4 9.0 1.2
///       5 6.0 2.2
///     EOF
/// ").unwrap();
///
/// let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 5, 100, 3);
/// let solution = solver.solve();
///
/// assert_eq!(solution.tour.len(), 5);
/// ```
pub struct AntColonySystem {
    tsp: Tsp,
    pheromones: Vec<Vec<f64>>,
    pheromone_scores: Vec<Vec<f64>>,
    heuristic_scores: Vec<Vec<f64>>,
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
    rng: StdRng,
    seed: u64,
}

impl AntColonySystem {
    /// Creates a new AntColonySystem with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to solve
    /// * `alpha` - Pheromone importance factor
    /// * `beta` - Heuristic information importance factor
    /// * `rho` - Pheromone evaporation rate
    /// * `q0` - Exploitation vs exploration factor
    /// * `num_ants` - Number of ants in the colony
    /// * `max_iterations` - Maximum number of iterations
    /// * `candidate_list_size` - Size of the candidate list for each city
    ///
    /// # Example
    ///
    /// ```
    /// use ibn_battuta::experimental::AntColonySystem;
    /// use ibn_battuta::TspBuilder;
    ///
    /// let tsp = TspBuilder::parse_str("
    ///     NAME : example
    ///     TYPE : TSP
    ///     DIMENSION : 5
    ///     EDGE_WEIGHT_TYPE: EUC_2D
    ///     NODE_COORD_SECTION
    ///       1 1.2 3.4
    ///       2 5.6 7.8
    ///       3 3.4 5.6
    ///       4 9.0 1.2
    ///       5 6.0 2.2
    ///     EOF
    /// ").unwrap();
    ///
    /// let acs = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 5, 100, 3);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        tsp: Tsp,
        alpha: f64,
        beta: f64,
        rho: f64,
        q0: f64,
        num_ants: usize,
        max_iterations: usize,
        candidate_list_size: usize,
    ) -> AntColonySystem {
        Self::with_options_and_seed(
            tsp,
            alpha,
            beta,
            rho,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
            rand::random(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_options_and_seed(
        tsp: Tsp,
        alpha: f64,
        beta: f64,
        rho: f64,
        q0: f64,
        num_ants: usize,
        max_iterations: usize,
        candidate_list_size: usize,
        seed: u64,
    ) -> AntColonySystem {
        let mut nn = NearestNeighbor::new(tsp.clone());
        let base_tour = nn.solve().length;
        let n = tsp.dim();
        let tau0 = 1.0 / (n as f64 * base_tour);

        let pheromones = vec![vec![tau0; n]; n];
        let pheromone_scores = vec![vec![tau0.powf(alpha); n]; n];
        let heuristic_scores = vec![vec![0.0; n]; n];

        let mut acs = AntColonySystem {
            tsp,
            pheromones,
            pheromone_scores,
            heuristic_scores,
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
            rng: StdRng::seed_from_u64(seed),
            seed,
        };

        acs.initialize_heuristic();
        acs.initialize_candidate_lists();
        acs
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn calculate_tour_cost(&self, tour: &[usize]) -> f64 {
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
                    self.heuristic_scores[i][j] = (1.0 / self.tsp.weight(i, j)).powf(self.beta);
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
            self.candidate_lists[i] = candidates
                .into_iter()
                .take(self.candidate_list_size)
                .map(|(j, _)| j)
                .collect();
        }
    }

    fn construct_solution(&mut self) -> Vec<usize> {
        let mut tour = vec![0; self.tsp.dim()];
        let mut visited = vec![false; self.tsp.dim()];

        tour[0] = self.rng.gen_range(0..self.tsp.dim());
        visited[tour[0]] = true;

        for i in 1..self.tsp.dim() {
            tour[i] = self.select_next_city(&tour[0..i], &visited);
            visited[tour[i]] = true;
            self.local_pheromone_update(&tour[i - 1..=i]);
        }

        // Close the tour
        self.local_pheromone_update(&[tour[self.tsp.dim() - 1], tour[0]]);

        tour
    }

    fn select_next_city(&mut self, partial_tour: &[usize], visited: &[bool]) -> usize {
        let current_city = partial_tour[partial_tour.len() - 1];

        if self.rng.gen::<f64>() < self.q0 {
            // Exploitation (choose best)
            self.select_best_city(current_city, visited)
        } else {
            // Exploration (probabilistic choice)
            self.select_probabilistic_city(current_city, visited)
        }
    }

    fn select_best_city(&self, current_city: usize, visited: &[bool]) -> usize {
        self.candidate_lists[current_city]
            .iter()
            .filter(|&&city| !visited[city])
            .max_by(|&&a, &&b| {
                let score_a =
                    self.pheromone_scores[current_city][a] * self.heuristic_scores[current_city][a];
                let score_b =
                    self.pheromone_scores[current_city][b] * self.heuristic_scores[current_city][b];
                score_a.partial_cmp(&score_b).unwrap()
            })
            .cloned()
            .unwrap_or_else(|| {
                // If all candidates are visited, choose the best among all unvisited cities
                (0..self.tsp.dim())
                    .filter(|&city| !visited[city])
                    .max_by(|&a, &b| {
                        let score_a = self.pheromone_scores[current_city][a]
                            * self.heuristic_scores[current_city][a];
                        let score_b = self.pheromone_scores[current_city][b]
                            * self.heuristic_scores[current_city][b];
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .unwrap()
            })
    }

    fn select_probabilistic_city(&mut self, current_city: usize, visited: &[bool]) -> usize {
        let mut total = 0.0;

        for &city in &self.candidate_lists[current_city] {
            if !visited[city] {
                total += self.pheromone_scores[current_city][city]
                    * self.heuristic_scores[current_city][city];
            }
        }

        if total == 0.0 {
            // If all candidates are visited, consider all unvisited cities
            for (city, &is_visited) in visited.iter().enumerate() {
                if !is_visited {
                    total += self.pheromone_scores[current_city][city]
                        * self.heuristic_scores[current_city][city];
                }
            }

            let mut random_value = self.rng.gen::<f64>() * total;
            for (city, &is_visited) in visited.iter().enumerate() {
                if !is_visited {
                    random_value -= self.pheromone_scores[current_city][city]
                        * self.heuristic_scores[current_city][city];
                    if random_value <= 0.0 {
                        return city;
                    }
                }
            }
        } else {
            let mut random_value = self.rng.gen::<f64>() * total;
            for &city in &self.candidate_lists[current_city] {
                if !visited[city] {
                    random_value -= self.pheromone_scores[current_city][city]
                        * self.heuristic_scores[current_city][city];
                    if random_value <= 0.0 {
                        return city;
                    }
                }
            }
        }

        // Fallback in case of floating-point precision issues
        visited.iter().position(|&v| !v).unwrap()
    }

    fn local_pheromone_update(&mut self, edge: &[usize]) {
        let (i, j) = (edge[0], edge[1]);
        self.pheromones[i][j] = (1.0 - self.rho) * self.pheromones[i][j] + self.rho * self.tau0;
        self.pheromones[j][i] = self.pheromones[i][j];
        self.pheromone_scores[i][j] = self.pheromones[i][j].powf(self.alpha);
        self.pheromone_scores[j][i] = self.pheromone_scores[i][j];
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
            self.pheromone_scores[from][to] = self.pheromones[from][to].powf(self.alpha);
            self.pheromone_scores[to][from] = self.pheromone_scores[from][to];
        }
    }

    fn update_best_solution(&mut self, tour: &mut Vec<usize>) {
        let cost = self.calculate_tour_cost(tour);
        if cost < self.best_cost {
            mem::swap(&mut self.best_tour, tour);
            self.best_cost = cost;
        }
    }
}

impl TspSolver for AntColonySystem {
    fn solve(&mut self) -> Solution {
        for _ in 0..self.max_iterations {
            for _ in 0..self.num_ants {
                let mut solution = self.construct_solution();
                self.update_best_solution(&mut solution);
            }
            self.global_pheromone_update();
        }

        Solution {
            tour: self.best_tour.clone(),
            length: self.best_cost,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        "ACS".to_string()
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
        let n = tsp.dim();
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, n, 1000, 3);

        let solution = solver.solve();

        assert_eq!(solution.tour.len(), n);
        assert!(solution.length > 0.0);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let n = tsp.dim();
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, n, 1000, 5);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), n);
        assert!(solution.length > 0.0);
    }

    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 10, 1000, 15);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), size);
        assert!(solution.length > 0.0);
    }

    #[test]
    fn test_st70() {
        let path = "data/tsplib/st70.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        test_instance(tsp);
    }

    #[test]
    fn test_berlin52() {
        let path = "data/tsplib/berlin52.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        test_instance(tsp);
    }

    #[test]
    fn test_pheromone_update() {
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
        let mut solver = AntColonySystem::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 5, 100, 3);

        solver.best_tour = vec![0, 1, 2, 3, 4];
        solver.best_cost = solver.calculate_tour_cost(&solver.best_tour);
        let pheromone_before = solver.pheromones[0][1];
        solver.global_pheromone_update();
        assert!(solver.pheromones[0][1] != pheromone_before);
    }

    #[test]
    fn uses_seeded_runs_deterministically() {
        let data = "
        NAME : example
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
        let mut lhs =
            AntColonySystem::with_options_and_seed(tsp.clone(), 0.1, 2.0, 0.1, 0.9, 5, 100, 3, 11);
        let mut rhs =
            AntColonySystem::with_options_and_seed(tsp, 0.1, 2.0, 0.1, 0.9, 5, 100, 3, 11);

        assert_eq!(lhs.seed(), 11);
        assert_eq!(lhs.solve(), rhs.solve());
    }
}
