use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use crate::NearestNeighbor;

/// The `TwoOpt` struct implements the 2-opt local search optimization algorithm for the Traveling Salesman Problem (TSP).
///
/// The 2-opt heuristic iteratively improves a given tour by removing two edges and reconnecting
/// the two paths in a different way to reduce the total tour cost.
///
/// # Example
///
/// ```
/// use ibn_battuta::TspBuilder;
/// use ibn_battuta::algorithms::TspSolver;
/// use ibn_battuta::algorithms::heuristic::local_search::two_opt::TwoOpt;
/// use ibn_battuta::NearestNeighbor;
///
/// let data = "
/// NAME : example
/// COMMENT : this is
/// COMMENT : a simple example
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
///
/// let tsp = TspBuilder::parse_str(data).unwrap();
///
/// let mut nn = NearestNeighbor::new(tsp.clone());
/// let base_tour = nn.solve().tour;
/// let mut solver = TwoOpt::from(tsp, base_tour, false);
/// let solution = solver.solve();
///
/// println!("Optimized tour: {:?}", solution.tour);
/// println!("Optimized cost: {:?}", solution.length);
/// ```
pub struct TwoOpt {
    tsp: Tsp,
    tour: Vec<usize>,
    cost: f64,
    verbose: bool,
    base_tour: Vec<usize>,
}

impl TwoOpt {
    /// Constructs a new `TwoOpt` solver with a tour initialized by the Nearest Neighbor heuristic.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to solve.
    ///
    /// # Returns
    ///
    /// A `TwoOpt` instance.
    pub fn new(tsp: Tsp) -> TwoOpt {
        let mut nn = NearestNeighbor::new(tsp.clone());
        let base_tour = nn.solve().tour;
        TwoOpt {
            tsp,
            tour: vec![],
            cost: 0.0,
            verbose: false,
            base_tour,
        }
    }

    /// Constructs a new `TwoOpt` solver with a given base tour and verbosity option.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to solve.
    /// * `base_tour` - The initial tour to optimize.
    /// * `verbose` - Whether to print details of each optimization step.
    ///
    /// # Returns
    ///
    /// A `TwoOpt` instance.
    pub fn from(tsp: Tsp, base_tour: Vec<usize>, verbose: bool) -> TwoOpt {
        TwoOpt {
            tsp,
            tour: vec![],
            cost: f64::INFINITY,
            verbose,
            base_tour,
        }
    }

    /// Performs a 2-opt swap on the tour between indices `i` and `k`.
    ///
    /// # Arguments
    ///
    /// * `tour` - The current tour.
    /// * `i` - The starting index of the segment to be reversed.
    /// * `k` - The ending index of the segment to be reversed.
    fn swap_2opt(tour: &mut Vec<usize>, i: usize, k: usize) {
        tour[i..=k].reverse();
    }

    /// Calculates the total cost of the current tour.
    ///
    /// # Returns
    ///
    /// The total tour cost.
    fn calculate_tour_cost(&self) -> f64 {
        let mut cost = 0.0;
        let len = self.tour.len();
        for i in 0..len {
            let from = self.tour[i];
            let to = self.tour[(i + 1) % len];
            cost += self.tsp.weight(from, to) as f64;
        }
        cost
    }

    /// Optimizes the tour using the 2-opt heuristic.
    ///
    /// Iteratively improves the current tour by attempting to swap edges
    /// to reduce the overall tour cost.
    pub fn optimize(&mut self) {
        let n = self.tour.len();
        if n < 3 {
            return;
        }

        let mut improved = true;

        while improved {
            improved = false;
            for i in 0..n - 2 {
                for j in i + 2..n {
                    let current_distance = self.tsp.weight(self.tour[i], self.tour[i + 1]) as f64
                        + self.tsp.weight(self.tour[j], self.tour[(j + 1) % n]) as f64;

                    let new_distance = self.tsp.weight(self.tour[i], self.tour[j]) as f64
                        + self.tsp.weight(self.tour[i + 1], self.tour[(j + 1) % n]) as f64;

                    if new_distance < current_distance {
                        Self::swap_2opt(&mut self.tour, i + 1, j);
                        self.cost = self.calculate_tour_cost();
                        improved = true;

                        if self.verbose {
                            println!(
                                "2OPT: Swapped edges ({} - {}) and ({} - {}), new cost: {}",
                                self.tour[i],
                                self.tour[i + 1],
                                self.tour[j],
                                self.tour[(j + 1) % n],
                                self.cost
                            );
                        }
                    }
                }
            }
        }
    }
}

impl TspSolver for TwoOpt {
    fn solve(&mut self) -> Solution {
        self.tour = self.base_tour.clone();
        self.cost = self.calculate_tour_cost();

        if self.verbose {
            println!("Initial tour: {:?}", self.tour);
            println!("Initial cost: {}", self.cost);
        }

        self.optimize();

        Solution {
            tour: self.tour.clone(),
            length: self.cost,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        format!("NN2Opt")
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::heuristic::local_search::two_opt::TwoOpt;
    use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
    use crate::algorithms::TspSolver;
    use crate::TspBuilder;

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let mut nn = NearestNeighbor::new(tsp.clone());
        let base_tour = nn.solve().tour;
        let mut solver = TwoOpt::from(tsp, base_tour, false);
        let solution = solver.solve();
        assert_eq!(solution.tour.len(), size);
    }

    #[test]
    fn two_opt_optimizes_tour() {
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
        let nn_sol = nn.solve();
        let mut solver = TwoOpt::from(tsp, nn_sol.tour, false);
        let solution = solver.solve();
        assert!(solution.length < nn_sol.length);
    }

    #[test]
    fn two_opt_handles_empty_tour() {
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

        let mut solver = TwoOpt::from(tsp, vec![], false);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 0);
        assert_eq!(solution.length, 0.0);
    }

    #[test]
    fn two_opt_handles_single_node_tour() {
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

        let mut solver = TwoOpt::from(tsp, vec![1], false);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 1);
        assert_eq!(solution.length, 0.0);
    }

    #[test]
    fn two_opt_handles_two_node_tour() {
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

        let mut solver = TwoOpt::from(tsp, vec![1, 2], false);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 2);
        assert!(solution.length > 0.0);
    }
}
