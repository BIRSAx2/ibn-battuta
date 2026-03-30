use crate::algorithms::utils::should_parallelize;
use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rayon::prelude::*;

/// A brute-force approach to solving the Traveling Salesman Problem (TSP).
///
/// The `BruteForce` algorithm computes all possible permutations of cities
/// and evaluates the cost of each tour. It then selects the tour with the minimum
/// cost as the solution.
///
/// # Attributes
/// * `tsp`: Reference to the `Tsp` instance.
/// * `best_tour`: Stores the best tour found so far.
/// * `best_cost`: Stores the cost of the best tour.
///
/// # Example
/// ```
/// use ibn_battuta::algorithms::TspSolver;
/// use ibn_battuta::TspBuilder;
/// use ibn_battuta::algorithms::exact::BruteForce;
///
/// let data = "
/// NAME : simple
/// TYPE : TSP
/// DIMENSION : 4
/// EDGE_WEIGHT_TYPE: EUC_2D
/// NODE_COORD_SECTION
///   1 0.0 0.0
///   2 0.0 1.0
///   3 1.0 1.0
///   4 1.0 0.0
/// EOF
/// ";
///
/// let tsp = TspBuilder::parse_str(data).unwrap();
/// let mut solver = BruteForce::new(&tsp);
/// let solution = solver.solve();
///
/// assert_eq!(solution.tour.len(), 4);
/// assert!((solution.length - 4.0).abs() < f64::EPSILON);
/// ```
pub struct BruteForce<'a> {
    tsp: &'a Tsp,
    best_tour: Vec<usize>,
    best_cost: f64,
}

impl TspSolver for BruteForce<'_> {
    /// Solves the TSP using brute force by computing all possible tours.
    ///
    /// Returns the optimal solution, including the best tour and its cost.
    fn solve(&mut self) -> Solution {
        if self.tsp.dim() <= 1 {
            self.best_tour = vec![0];
            self.best_cost = 0.0;
            return Solution::new(self.best_tour.clone(), self.best_cost);
        }

        let starting_branches: Vec<usize> = (1..self.tsp.dim()).collect();
        if should_parallelize(starting_branches.len()) {
            if let Some((best_tour, best_cost)) = starting_branches
                .into_par_iter()
                .map(|next_city| {
                    let mut tour = vec![0, next_city];
                    self.solve_recursive_collect(&mut tour, self.tsp.weight(0, next_city))
                })
                .filter_map(|candidate| candidate)
                .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
            {
                self.best_tour = best_tour;
                self.best_cost = best_cost;
            }
        } else {
            let mut tour = vec![0];
            self.solve_recursive(&mut tour, 0.0);
        }

        Solution::new(self.best_tour.clone(), self.best_cost)
    }

    /// Returns the best tour found after solving.
    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    /// Returns the cost between two cities.
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}

impl<'a> BruteForce<'a> {
    /// Creates a new `BruteForce` solver for the given TSP instance.
    ///
    /// # Arguments
    /// * `tsp` - Reference to the `Tsp` problem.
    pub fn new(tsp: &'a Tsp) -> Self {
        BruteForce {
            tsp,
            best_tour: vec![],
            best_cost: f64::INFINITY, // Initialize with infinity cost
        }
    }

    /// Recursively explores all possible tours and updates the best one found.
    ///
    /// # Arguments
    /// * `tour` - The current partial tour being explored.
    /// * `cost` - The current cost of the partial tour.
    fn solve_recursive(&mut self, tour: &mut [usize], cost: f64) {
        if let Some((best_tour, best_cost)) = self.solve_recursive_collect(tour, cost) {
            self.best_tour = best_tour;
            self.best_cost = best_cost;
        }
    }

    fn solve_recursive_collect(&self, tour: &mut [usize], cost: f64) -> Option<(Vec<usize>, f64)> {
        if tour.len() == self.tsp.dim() {
            let last = tour.last().unwrap();
            let cost = cost + self.tsp.weight(*last, tour[0]);
            Some((tour.to_vec(), cost))
        } else {
            let mut best: Option<(Vec<usize>, f64)> = None;
            for i in 0..self.tsp.dim() {
                if !tour.contains(&i) {
                    let mut new_tour = tour.to_vec();
                    new_tour.push(i);
                    let new_cost = cost + self.tsp.weight(*tour.last().unwrap(), i);
                    if let Some(candidate) = self.solve_recursive_collect(&mut new_tour, new_cost) {
                        if best
                            .as_ref()
                            .map(|(_, best_cost)| candidate.1 < *best_cost)
                            .unwrap_or(true)
                        {
                            best = Some(candidate);
                        }
                    }
                }
            }
            best
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::exact::brute_force::BruteForce;
    use crate::algorithms::TspSolver;
    use crate::TspBuilder;

    #[test]
    fn solves_simple_tsp_with_brute_force() {
        let data = "
    NAME : simple
    TYPE : TSP
    DIMENSION : 4
    EDGE_WEIGHT_TYPE: EUC_2D
    NODE_COORD_SECTION
      1 0.0 0.0
      2 0.0 1.0
      3 1.0 1.0
      4 1.0 0.0
    EOF
    ";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = BruteForce::new(&tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 4);
        assert!((solution.length - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_single_node_with_brute_force() {
        let data = "
    NAME : single
    TYPE : TSP
    DIMENSION : 1
    EDGE_WEIGHT_TYPE: EUC_2D
    NODE_COORD_SECTION
      1 0.0 0.0
    EOF
    ";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = BruteForce::new(&tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 1);
        assert!((solution.length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_two_nodes_with_brute_force() {
        let data = "
    NAME : two_nodes
    TYPE : TSP
    DIMENSION : 2
    EDGE_WEIGHT_TYPE: EUC_2D
    NODE_COORD_SECTION
      1 0.0 0.0
      2 1.0 0.0
    EOF
    ";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = BruteForce::new(&tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 2);
        assert!((solution.length - 2.0).abs() < f64::EPSILON);
    }
}
