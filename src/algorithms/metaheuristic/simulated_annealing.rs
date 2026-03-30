//! This module implements a Simulated Annealing algorithm for solving the Traveling Salesman Problem (TSP).
//!
//! The Simulated Annealing algorithm is a probabilistic technique for approximating the global optimum
//! of a given function. It is often used when the search space is discrete.

use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::f64;

/// Represents a Simulated Annealing solver for the Traveling Salesman Problem.
pub struct SimulatedAnnealing {
    tsp: Tsp,
    fire: Vec<usize>,
    fire_length: f64,
    best_path: Vec<usize>,
    best_length: f64,
    t0: f64,
    tend: f64,
    rate: f64,
    iter_x: Vec<usize>,
    iter_y: Vec<f64>,
    rng: StdRng,
    seed: u64,
}

impl SimulatedAnnealing {
    /// Creates a new instance of SimulatedAnnealing.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to solve.
    ///
    /// # Returns
    ///
    /// A new SimulatedAnnealing instance.
    ///
    /// # Example
    ///
    /// ```
    /// use ibn_battuta::experimental::SimulatedAnnealing;
    /// use ibn_battuta::TspBuilder;
    ///
    /// let data = "
    /// NAME : example
    /// COMMENT : Simple example
    /// TYPE : TSP
    /// DIMENSION : 5
    /// EDGE_WEIGHT_TYPE: EUC_2D
    /// NODE_COORD_SECTION
    ///   1 1.2 3.4
    ///   2 5.6 7.8
    ///   3 3.4 5.6
    ///   4 9.0 1.2
    ///   5 6.0 2.2
    /// EOF
    /// ";
    /// let tsp = TspBuilder::parse_str(data).unwrap();
    /// let sa = SimulatedAnnealing::new(tsp);
    /// ```
    pub fn new(tsp: Tsp) -> Self {
        Self::with_seed(tsp, rand::random())
    }

    /// Creates a new SimulatedAnnealing instance with a deterministic seed.
    pub fn with_seed(tsp: Tsp, seed: u64) -> Self {
        let num_city = tsp.dim();
        let mut sa = SimulatedAnnealing {
            tsp: tsp.clone(),
            fire: vec![],
            fire_length: f64::MAX,
            best_path: vec![],
            best_length: f64::MAX,
            t0: 4000.0,
            tend: 1e-3,
            rate: 0.9997,
            iter_x: vec![0],
            iter_y: vec![0.0],
            rng: StdRng::seed_from_u64(seed),
            seed,
        };

        let fire = sa.greedy_init(&tsp, 100, num_city);
        let init_pathlen = sa.compute_pathlen(&fire, &tsp);
        sa.fire = fire.clone();
        sa.fire_length = init_pathlen;
        sa.best_path = fire;
        sa.best_length = init_pathlen;
        sa.iter_y[0] = init_pathlen;
        sa
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Initializes the solution using a greedy approach.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance.
    /// * `num_total` - The number of initial solutions to generate.
    /// * `num_city` - The number of cities in the TSP.
    ///
    /// # Returns
    ///
    /// A vector representing the initial solution.
    fn greedy_init(&mut self, tsp: &Tsp, num_total: usize, num_city: usize) -> Vec<usize> {
        let mut result = Vec::new();

        for _ in 0..num_total {
            let mut rest: Vec<usize> = (0..num_city).collect();
            let mut current = if result.len() < num_city {
                result.len()
            } else {
                self.rng.gen_range(0..num_city)
            };

            let mut result_one = vec![current];
            rest.retain(|&x| x != current);

            while !rest.is_empty() {
                let (tmp_choose, _) = rest
                    .iter()
                    .map(|&x| (x, self.cost(current, x)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();

                current = tmp_choose;
                result_one.push(tmp_choose);
                rest.retain(|&x| x != tmp_choose);
            }

            result.push(result_one);
        }

        let path_lens: Vec<f64> = result
            .iter()
            .map(|path| self.compute_pathlen(path, tsp))
            .collect();

        result[path_lens
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0]
            .clone()
    }

    /// Computes the total path length for a given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to compute the length for.
    /// * `_tsp` - The TSP instance (unused in this implementation).
    ///
    /// # Returns
    ///
    /// The total length of the path.
    fn compute_pathlen(&self, path: &[usize], _tsp: &Tsp) -> f64 {
        let mut result = self.cost(*path.last().unwrap(), path[0]);
        for i in 0..path.len() - 1 {
            result += self.cost(path[i], path[i + 1]);
        }
        result
    }

    fn sample_move(&mut self) -> (usize, usize) {
        let len = self.fire.len();
        let a = self.rng.gen_range(0..len);
        let b = self.rng.gen_range(0..len);
        (a.min(b), a.max(b))
    }

    fn reversed_segment_length(&self, start: usize, end: usize) -> f64 {
        if start == end {
            return self.fire_length;
        }

        let path = &self.fire;
        let len = path.len();
        if start == 0 && end + 1 == len {
            return self.fire_length;
        }

        let prev = if start == 0 { len - 1 } else { start - 1 };
        let next = (end + 1) % len;

        let mut removed = self.cost(path[prev], path[start]) + self.cost(path[end], path[next]);
        let mut added = self.cost(path[prev], path[end]) + self.cost(path[start], path[next]);

        for idx in start..end {
            removed += self.cost(path[idx], path[idx + 1]);
            added += self.cost(path[idx + 1], path[idx]);
        }

        self.fire_length - removed + added
    }

    /// Evaluates a new solution and decides whether to accept it.
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the reversed segment.
    /// * `end` - The end of the reversed segment.
    /// * `temp` - The current temperature.
    ///
    /// # Returns
    ///
    /// `true` if the move was accepted.
    fn eval_fire(&mut self, start: usize, end: usize, temp: f64) -> bool {
        let len2 = self.reversed_segment_length(start, end);
        let dc = len2 - self.fire_length;
        let accept = dc <= 0.0 || self.rng.gen::<f64>() <= f64::exp(-dc / temp);

        if accept {
            self.fire[start..=end].reverse();
            self.fire_length = len2;
            true
        } else {
            false
        }
    }

    /// Runs the Simulated Annealing algorithm.
    ///
    /// # Returns
    ///
    /// A tuple containing the best path length and the best path found.
    pub fn sa(&mut self) -> (f64, Vec<usize>) {
        let mut count = 0;
        let mut t = self.t0;

        while t > self.tend {
            count += 1;
            let (start, end) = self.sample_move();
            if self.eval_fire(start, end, t) && self.fire_length < self.best_length {
                self.best_path = self.fire.clone();
                self.best_length = self.fire_length;
            }

            t *= self.rate;

            self.iter_x.push(count);
            self.iter_y.push(self.best_length);
        }

        (self.best_length, self.best_path.clone())
    }
}

impl TspSolver for SimulatedAnnealing {
    /// Solves the TSP using Simulated Annealing.
    ///
    /// # Returns
    ///
    /// A `Solution` struct containing the best tour and its total length.
    fn solve(&mut self) -> Solution {
        let (best_length, best_path) = self.sa();
        Solution {
            tour: best_path,
            length: best_length,
        }
    }

    /// Returns the current tour.
    ///
    /// # Returns
    ///
    /// A vector representing the current tour.
    fn tour(&self) -> Vec<usize> {
        self.fire.clone()
    }

    /// Computes the cost between two cities.
    ///
    /// # Arguments
    ///
    /// * `from` - The index of the starting city.
    /// * `to` - The index of the ending city.
    ///
    /// # Returns
    ///
    /// The cost (distance) between the two cities.
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    /// Returns the name of the solver.
    ///
    /// # Returns
    ///
    /// A string representing the name of the solver.
    fn format_name(&self) -> String {
        "SA".to_string()
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
        let mut solver = SimulatedAnnealing::new(tsp.clone());
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 5);
        assert!(solution.length > 0.0);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/bier127.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let sol = test_instance(tsp);
        let best_known = 118282.0;
        let gap = (sol.length - best_known) / best_known;
        assert!(gap < 0.1, "Gap is too large: {:.2}%", gap * 100.0);
    }

    fn test_instance(tsp: Tsp) -> Solution {
        let size = tsp.dim();
        let mut solver = SimulatedAnnealing::new(tsp);
        let solution = solver.solve();
        assert_eq!(solution.tour.len(), size);
        solution
    }

    #[test]
    fn test_greedy_init() {
        let data = "
        NAME : example
        TYPE : TSP
        DIMENSION : 5
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 1.0 1.0
          2 2.0 2.0
          3 3.0 3.0
          4 4.0 4.0
          5 5.0 5.0
        EOF
        ";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut sa = SimulatedAnnealing::new(tsp.clone());
        let init_path = sa.greedy_init(&tsp, 10, 5);

        assert_eq!(init_path.len(), 5);
        assert!(init_path.iter().all(|&x| x < 5));
    }

    #[test]
    fn test_reversed_segment_length_matches_full_recompute() {
        let data = "
        NAME : example
        TYPE : TSP
        DIMENSION : 5
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 1.0 1.0
          2 2.0 2.0
          3 3.0 3.0
          4 4.0 4.0
          5 5.0 5.0
        EOF
        ";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut sa = SimulatedAnnealing::new(tsp);
        sa.fire = vec![0, 1, 2, 3, 4];
        sa.fire_length = sa.compute_pathlen(&sa.fire, &sa.tsp);

        let start = 1;
        let end = 3;
        let candidate_len = sa.reversed_segment_length(start, end);

        let mut recomputed = sa.fire.clone();
        recomputed[start..=end].reverse();

        assert_eq!(recomputed.len(), 5);
        assert!(
            (candidate_len - sa.compute_pathlen(&recomputed, &sa.tsp)).abs() < 1e-9,
            "local segment evaluation must match full path recomputation"
        );
    }

    #[test]
    fn uses_seeded_runs_deterministically() {
        let data = "
        NAME : example
        TYPE : TSP
        DIMENSION : 5
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 1.0 1.0
          2 2.0 2.0
          3 3.0 3.0
          4 4.0 4.0
          5 5.0 5.0
        EOF
        ";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut lhs = SimulatedAnnealing::with_seed(tsp.clone(), 42);
        let mut rhs = SimulatedAnnealing::with_seed(tsp, 42);

        assert_eq!(lhs.seed(), 42);
        assert_eq!(lhs.solve(), rhs.solve());
    }
}
