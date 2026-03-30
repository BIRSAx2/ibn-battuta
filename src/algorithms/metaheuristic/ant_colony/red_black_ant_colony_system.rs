use crate::algorithms::TspSolver;
use crate::parser::Tsp;
use crate::{NearestNeighbor, Solution};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::{cmp::Ordering, f64, mem};

/// Red-Black Ant Colony System (RB-ACS) solver for the Traveling Salesman Problem.
///
/// This implementation is based on the paper "An Improved ACS Algorithm for the Solutions of
/// Larger TSP Problems" by Md. Rakib Hassan, Md. Kamrul Hasan and M.M.A Hashem.
///
/// # Example
///
/// ```no_run
/// use ibn_battuta::experimental::RedBlackACS;
/// use ibn_battuta::TspBuilder;
/// use ibn_battuta::TspSolver;
///
/// let tsp = TspBuilder::parse_path("path/to/tsp/file.tsp").unwrap();
/// let mut solver = RedBlackACS::new(tsp, 1.0, 2.0, 0.1, 0.2, 0.9, 20, 1000, 15);
/// let solution = solver.solve();
/// println!("Best tour: {:?}", solution.tour);
/// println!("Tour length: {}", solution.length);
/// ```
pub struct RedBlackACS {
    tsp: Tsp,
    pheromones_red: Vec<Vec<f64>>,
    pheromones_black: Vec<Vec<f64>>,
    pheromone_scores_red: Vec<Vec<f64>>,
    pheromone_scores_black: Vec<Vec<f64>>,
    heuristic_scores: Vec<Vec<f64>>,
    candidate_lists: Vec<Vec<usize>>,
    best_tour_red: Vec<usize>,
    best_tour_black: Vec<usize>,
    best_cost_red: f64,
    best_cost_black: f64,
    alpha: f64,
    beta: f64,
    rho_red: f64,
    rho_black: f64,
    tau0: f64,
    q0: f64,
    num_ants: usize,
    max_iterations: usize,
    candidate_list_size: usize,
    rng: StdRng,
    seed: u64,
}

impl RedBlackACS {
    /// Creates a new RedBlackACS solver.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to solve
    /// * `alpha` - Pheromone influence parameter
    /// * `beta` - Heuristic influence parameter
    /// * `rho_red` - Pheromone evaporation rate for red ants
    /// * `rho_black` - Pheromone evaporation rate for black ants
    /// * `q0` - Exploitation vs. exploration parameter
    /// * `num_ants` - Number of ants per color
    /// * `max_iterations` - Maximum number of iterations
    /// * `candidate_list_size` - Size of the candidate list for each city
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ibn_battuta::experimental::RedBlackACS;
    /// use ibn_battuta::TspBuilder;
    /// let tsp = TspBuilder::parse_path("path/to/tsp/file.tsp").unwrap();
    /// let solver = RedBlackACS::new(tsp, 1.0, 2.0, 0.1, 0.2, 0.9, 20, 1000, 15);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tsp: Tsp,
        alpha: f64,
        beta: f64,
        rho_red: f64,
        rho_black: f64,
        q0: f64,
        num_ants: usize,
        max_iterations: usize,
        candidate_list_size: usize,
    ) -> RedBlackACS {
        Self::new_with_seed(
            tsp,
            alpha,
            beta,
            rho_red,
            rho_black,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
            rand::random(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_seed(
        tsp: Tsp,
        alpha: f64,
        beta: f64,
        rho_red: f64,
        rho_black: f64,
        q0: f64,
        num_ants: usize,
        max_iterations: usize,
        candidate_list_size: usize,
        seed: u64,
    ) -> RedBlackACS {
        let dim = tsp.dim();
        let mut pheromones_red = vec![vec![0.0; dim]; dim];
        let mut pheromones_black = vec![vec![0.0; dim]; dim];
        let mut pheromone_scores_red = vec![vec![0.0; dim]; dim];
        let mut pheromone_scores_black = vec![vec![0.0; dim]; dim];
        let heuristic_scores = vec![vec![0.0; dim]; dim];

        // Calculate tau0 based on nearest neighbor heuristic
        let nn_solution = NearestNeighbor::new(tsp.clone()).solve();
        let nn_tour = nn_solution.tour;
        let nn_tour_length = nn_solution.length;
        let tau0 = 1.0 / (dim as f64 * nn_tour_length);
        let mut reverse_nn_tour = nn_tour.clone();
        reverse_nn_tour.reverse();

        // Initialize pheromones inversely proportional to edge weights
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    let cost = tsp.weight(i, j);
                    pheromones_red[i][j] = 100.0 / cost;
                    pheromones_black[i][j] = 100.0 / cost;
                    pheromone_scores_red[i][j] = pheromones_red[i][j].powf(alpha);
                    pheromone_scores_black[i][j] = pheromones_black[i][j].powf(alpha);
                }
            }
        }

        let mut rb_acs = RedBlackACS {
            tsp,
            pheromones_red,
            pheromones_black,
            pheromone_scores_red,
            pheromone_scores_black,
            heuristic_scores,
            candidate_lists: vec![],
            best_tour_red: nn_tour,
            best_tour_black: reverse_nn_tour,
            best_cost_red: nn_tour_length,
            best_cost_black: nn_tour_length,
            alpha,
            beta,
            rho_red,
            rho_black,
            tau0,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
            rng: StdRng::seed_from_u64(seed),
            seed,
        };

        rb_acs.initialize_heuristic();
        rb_acs.initialize_candidate_lists();
        rb_acs
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn best_group_tours(&self) -> [(Vec<usize>, f64); 2] {
        [
            (self.best_tour_red.clone(), self.best_cost_red),
            (self.best_tour_black.clone(), self.best_cost_black),
        ]
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

    fn construct_solution(&mut self, is_red: bool) -> Vec<usize> {
        let mut tour = vec![0; self.tsp.dim()];
        let mut visited = vec![false; self.tsp.dim()];

        tour[0] = self.rng.gen_range(0..self.tsp.dim());
        visited[tour[0]] = true;

        for i in 1..self.tsp.dim() {
            tour[i] = self.select_next_city(&tour[0..i], &visited, is_red);
            visited[tour[i]] = true;
            self.local_pheromone_update(&tour[i - 1..=i], is_red);
        }

        // Close the tour
        self.local_pheromone_update(&[tour[self.tsp.dim() - 1], tour[0]], is_red);

        tour
    }

    fn select_next_city(
        &mut self,
        partial_tour: &[usize],
        visited: &[bool],
        is_red: bool,
    ) -> usize {
        let current_city = partial_tour[partial_tour.len() - 1];

        if self.rng.gen::<f64>() < self.q0 {
            let pheromone_scores = if is_red {
                &self.pheromone_scores_red
            } else {
                &self.pheromone_scores_black
            };
            self.select_best_city(current_city, visited, pheromone_scores)
        } else {
            self.select_probabilistic_city(current_city, visited, is_red)
        }
    }

    fn select_best_city(
        &self,
        current_city: usize,
        visited: &[bool],
        pheromone_scores: &[Vec<f64>],
    ) -> usize {
        self.candidate_lists[current_city]
            .iter()
            .filter(|&&city| !visited[city])
            .max_by(|&&a, &&b| {
                let score_a =
                    pheromone_scores[current_city][a] * self.heuristic_scores[current_city][a];
                let score_b =
                    pheromone_scores[current_city][b] * self.heuristic_scores[current_city][b];
                score_a.partial_cmp(&score_b).unwrap()
            })
            .cloned()
            .unwrap_or_else(|| {
                (0..self.tsp.dim())
                    .filter(|&city| !visited[city])
                    .max_by(|&a, &b| {
                        let score_a = pheromone_scores[current_city][a]
                            * self.heuristic_scores[current_city][a];
                        let score_b = pheromone_scores[current_city][b]
                            * self.heuristic_scores[current_city][b];
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .unwrap()
            })
    }

    fn select_probabilistic_city(
        &mut self,
        current_city: usize,
        visited: &[bool],
        is_red: bool,
    ) -> usize {
        let pheromone_scores = if is_red {
            &self.pheromone_scores_red
        } else {
            &self.pheromone_scores_black
        };
        let mut total = 0.0;

        for &city in &self.candidate_lists[current_city] {
            if !visited[city] {
                total += pheromone_scores[current_city][city]
                    * self.heuristic_scores[current_city][city];
            }
        }

        if total == 0.0 {
            for (city, &is_visited) in visited.iter().enumerate() {
                if !is_visited {
                    total += pheromone_scores[current_city][city]
                        * self.heuristic_scores[current_city][city];
                }
            }

            let mut random_value = self.rng.gen::<f64>() * total;
            for (city, &is_visited) in visited.iter().enumerate() {
                if !is_visited {
                    random_value -= pheromone_scores[current_city][city]
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
                    random_value -= pheromone_scores[current_city][city]
                        * self.heuristic_scores[current_city][city];
                    if random_value <= 0.0 {
                        return city;
                    }
                }
            }
        }

        visited.iter().position(|&v| !v).unwrap()
    }

    fn local_pheromone_update(&mut self, edge: &[usize], is_red: bool) {
        let (i, j) = (edge[0], edge[1]);
        let (pheromones, pheromone_scores, rho) = if is_red {
            (
                &mut self.pheromones_red,
                &mut self.pheromone_scores_red,
                self.rho_red,
            )
        } else {
            (
                &mut self.pheromones_black,
                &mut self.pheromone_scores_black,
                self.rho_black,
            )
        };
        pheromones[i][j] = (1.0 - rho) * pheromones[i][j] + rho * self.tau0;
        pheromones[j][i] = pheromones[i][j];
        pheromone_scores[i][j] = pheromones[i][j].powf(self.alpha);
        pheromone_scores[j][i] = pheromone_scores[i][j];
    }

    fn global_pheromone_update(&mut self) {
        let deposit_red = 1.0 / self.best_cost_red;
        let deposit_black = 1.0 / self.best_cost_black;

        // Update pheromones for the best tours in each group.
        for i in 0..self.best_tour_red.len() {
            let from = self.best_tour_red[i];
            let to = self.best_tour_red[(i + 1) % self.best_tour_red.len()];
            self.pheromones_red[from][to] =
                (1.0 - self.rho_red) * self.pheromones_red[from][to] + self.rho_red * deposit_red;
            self.pheromones_red[to][from] = self.pheromones_red[from][to];
            self.pheromone_scores_red[from][to] = self.pheromones_red[from][to].powf(self.alpha);
            self.pheromone_scores_red[to][from] = self.pheromone_scores_red[from][to];
        }

        for i in 0..self.best_tour_black.len() {
            let from = self.best_tour_black[i];
            let to = self.best_tour_black[(i + 1) % self.best_tour_black.len()];
            self.pheromones_black[from][to] = (1.0 - self.rho_black)
                * self.pheromones_black[from][to]
                + self.rho_black * deposit_black;
            self.pheromones_black[to][from] = self.pheromones_black[from][to];
            self.pheromone_scores_black[from][to] =
                self.pheromones_black[from][to].powf(self.alpha);
            self.pheromone_scores_black[to][from] = self.pheromone_scores_black[from][to];
        }
    }

    fn update_best_solution(&mut self, tour: &mut Vec<usize>, is_red: bool) {
        let cost = self.calculate_tour_cost(tour);
        if is_red {
            if cost < self.best_cost_red {
                mem::swap(&mut self.best_tour_red, tour);
                self.best_cost_red = cost;
            }
        } else if cost < self.best_cost_black {
            mem::swap(&mut self.best_tour_black, tour);
            self.best_cost_black = cost;
        }
    }

    fn calculate_tour_cost(&self, tour: &[usize]) -> f64 {
        tour.windows(2)
            .map(|w| self.tsp.weight(w[0], w[1]))
            .sum::<f64>()
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
            length: self.best_cost_red.min(self.best_cost_black),
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
        "RBACS".to_string()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::TspBuilder;

    fn create_simple_tsp() -> Tsp {
        let data = "
        NAME : simple_tsp
        COMMENT : A simple 5-city TSP for testing
        TYPE : TSP
        DIMENSION : 5
        EDGE_WEIGHT_TYPE : EUC_2D
        NODE_COORD_SECTION
        1 0 0
        2 1 0
        3 1 1
        4 0 1
        5 0.5 0.5
        EOF
        ";
        TspBuilder::parse_str(data).unwrap()
    }

    #[test]
    fn test_initialization() {
        let tsp = create_simple_tsp();
        let solver = RedBlackACS::new(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3);

        assert_eq!(solver.tsp.dim(), 5);
        assert_eq!(solver.pheromones_red.len(), 5);
        assert_eq!(solver.pheromones_black.len(), 5);
        assert_eq!(solver.heuristic_scores.len(), 5);
        assert_eq!(solver.candidate_lists.len(), 5);

        // Check that pheromones are initialized correctly
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(solver.pheromones_red[i][j] > 0.0);
                    assert!(solver.pheromones_black[i][j] > 0.0);
                    assert_eq!(solver.pheromones_red[i][j], solver.pheromones_black[i][j]);
                }
            }
        }
    }

    #[test]
    fn test_construct_solution() {
        let tsp = create_simple_tsp();
        let mut solver = RedBlackACS::new(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3);

        let red_solution = solver.construct_solution(true);
        let black_solution = solver.construct_solution(false);

        assert_eq!(red_solution.len(), 5);
        assert_eq!(black_solution.len(), 5);

        // Check that all cities are visited exactly once
        for solution in &[red_solution, black_solution] {
            let mut visited = [false; 5];
            for &city in solution {
                assert!(!visited[city], "City {} visited more than once", city);
                visited[city] = true;
            }
            assert!(visited.iter().all(|&v| v), "Not all cities were visited");
        }
    }

    #[test]
    fn test_update_best_solution() {
        let tsp = create_simple_tsp();
        let mut solver = RedBlackACS::new(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3);

        let red_tour = vec![0, 1, 2, 3, 4];
        let black_tour = vec![0, 2, 4, 1, 3];
        let initial_red_cost = solver.best_cost_red;
        let initial_black_cost = solver.best_cost_black;

        solver.update_best_solution(&mut red_tour.clone(), true);
        solver.update_best_solution(&mut black_tour.clone(), false);

        let red_cost = solver.calculate_tour_cost(&red_tour);
        let black_cost = solver.calculate_tour_cost(&black_tour);

        assert!(solver.best_cost_red <= initial_red_cost);
        assert!(solver.best_cost_black <= initial_black_cost);
        assert_eq!(solver.best_cost_red, solver.calculate_tour_cost(&solver.best_tour_red));
        assert_eq!(
            solver.best_cost_black,
            solver.calculate_tour_cost(&solver.best_tour_black)
        );
        assert!(solver.best_cost_red <= red_cost);
        assert!(solver.best_cost_black <= black_cost);
    }

    #[test]
    fn test_solve_simple_tsp() {
        let tsp = create_simple_tsp();
        let mut solver = RedBlackACS::new(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3);

        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 5);
        assert!(solution.length > 0.0);

        // Check if the solution is a valid tour
        let mut visited = [false; 5];
        for &city in &solution.tour {
            assert!(!visited[city], "City {} visited more than once", city);
            visited[city] = true;
        }
        assert!(visited.iter().all(|&v| v), "Not all cities were visited");

        // Verify the tour cost
        let calculated_cost = solver.calculate_tour_cost(&solution.tour);
        assert!(
            (calculated_cost - solution.length).abs() < 1e-6,
            "Reported tour cost doesn't match calculated cost"
        );
    }

    #[test]
    fn test_local_pheromone_update() {
        let tsp = create_simple_tsp();
        let mut solver = RedBlackACS::new(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3);

        let initial_red_pheromone = solver.pheromones_red[0][1];
        let initial_black_pheromone = solver.pheromones_black[0][1];

        solver.local_pheromone_update(&[0, 1], true);
        solver.local_pheromone_update(&[0, 1], false);

        assert_ne!(solver.pheromones_red[0][1], initial_red_pheromone);
        assert_ne!(solver.pheromones_black[0][1], initial_black_pheromone);
        assert_eq!(solver.pheromones_red[0][1], solver.pheromones_red[1][0]);
        assert_eq!(solver.pheromones_black[0][1], solver.pheromones_black[1][0]);
    }

    #[test]
    fn test_global_pheromone_update() {
        let tsp = create_simple_tsp();
        let mut solver = RedBlackACS::new(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3);

        solver.best_tour_red = vec![0, 1, 2, 3, 4];
        solver.best_tour_black = vec![0, 2, 4, 1, 3];
        solver.best_cost_red = solver.calculate_tour_cost(&solver.best_tour_red);
        solver.best_cost_black = solver.calculate_tour_cost(&solver.best_tour_black);

        let initial_red_pheromone = solver.pheromones_red[0][1];
        let initial_black_pheromone = solver.pheromones_black[0][2];

        solver.global_pheromone_update();

        assert_ne!(solver.pheromones_red[0][1], initial_red_pheromone);
        assert_ne!(solver.pheromones_black[0][2], initial_black_pheromone);
    }

    #[test]
    fn uses_seeded_runs_deterministically() {
        let tsp = create_simple_tsp();
        let mut lhs =
            RedBlackACS::new_with_seed(tsp.clone(), 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3, 17);
        let mut rhs = RedBlackACS::new_with_seed(tsp, 1.0, 2.0, 0.1, 0.2, 0.9, 5, 100, 3, 17);

        assert_eq!(lhs.seed(), 17);
        assert_eq!(lhs.solve(), rhs.solve());
    }
}
