use crate::experimental::SimulatedAnnealing;
use crate::{Solution, Tsp, TspSolver, TwoOpt};

/// A hybrid solver that combines Simulated Annealing and 2-Opt local search
///
/// # Example
///
/// ```
/// use ibn_battuta::experimental::SA2Opt;
/// use ibn_battuta::{Tsp, TspSolver, TspBuilder};
///
/// let tsp_data = "
/// NAME : example
/// COMMENT : Small example TSP
/// TYPE : TSP
/// DIMENSION : 5
/// EDGE_WEIGHT_TYPE : EUC_2D
/// NODE_COORD_SECTION
/// 1 0 0
/// 2 1 0
/// 3 1 1
/// 4 0 1
/// 5 0.5 0.5
/// EOF
/// ";
///
/// let tsp = TspBuilder::parse_str(tsp_data).unwrap();
/// let mut solver = SA2Opt::new(tsp);
/// let solution = solver.solve();
///
/// assert_eq!(solution.tour.len(), 5);
/// ```
pub struct SA2Opt {
    tsp: Tsp,
    sa: SimulatedAnnealing,
    base_solution: Solution,
    last_solution: Solution,
}

impl SA2Opt {
    /// Creates a new SA2Opt solver instance
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to be solved
    ///
    /// # Example
    ///
    /// ```
    /// use ibn_battuta::experimental::SA2Opt;
    /// use ibn_battuta::TspBuilder;
    ///
    /// let tsp_data = "
    /// NAME : example
    /// COMMENT : Small example TSP
    /// TYPE : TSP
    /// DIMENSION : 3
    /// EDGE_WEIGHT_TYPE : EUC_2D
    /// NODE_COORD_SECTION
    /// 1 0 0
    /// 2 1 0
    /// 3 0 1
    /// EOF
    /// ";
    ///
    /// let tsp = TspBuilder::parse_str(tsp_data).unwrap();
    /// let solver = SA2Opt::new(tsp);
    /// ```
    pub fn new(tsp: Tsp) -> SA2Opt {
        Self::new_with_seed(tsp, rand::random())
    }

    pub fn new_with_seed(tsp: Tsp, seed: u64) -> SA2Opt {
        let sa = SimulatedAnnealing::with_seed(tsp.clone(), seed);

        SA2Opt {
            tsp,
            sa,
            base_solution: Solution::default(),
            last_solution: Solution::default(),
        }
    }

    pub fn seed(&self) -> u64 {
        self.sa.seed()
    }

    pub fn base_solution(&self) -> Solution {
        self.base_solution.clone()
    }
}

impl TspSolver for SA2Opt {
    fn solve(&mut self) -> Solution {
        self.base_solution = self.sa.solve();
        self.last_solution =
            TwoOpt::from(self.tsp.clone(), self.base_solution.tour.clone(), false).solve();
        Solution {
            tour: self.last_solution.tour.clone(),
            length: self.last_solution.length,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.last_solution.tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        "SA2Opt".to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::experimental::SA2Opt;
    use crate::{Tsp, TspBuilder, TspSolver};
    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        test_instance(tsp);
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

    #[test]
    fn test_custom_small_instance() {
        let tsp_data = "
        NAME : custom_small
        COMMENT : Custom small TSP instance
        TYPE : TSP
        DIMENSION : 6
        EDGE_WEIGHT_TYPE : EUC_2D
        NODE_COORD_SECTION
        1 0 0
        2 1 0
        3 2 0
        4 2 1
        5 1 1
        6 0 1
        EOF
        ";

        let tsp = TspBuilder::parse_str(tsp_data).unwrap();
        test_instance(tsp);
    }
    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();
        let mut solver = SA2Opt::new(tsp);
        let solution = solver.solve();

        assert_eq!(
            solution.tour.len(),
            size,
            "Tour length should match the number of cities"
        );
        assert!(
            solution.length > 0.0,
            "Total tour length should be positive"
        );

        // Check if the tour is a valid permutation
        let mut visited = vec![false; size];
        for &city in &solution.tour {
            assert!(!visited[city], "Each city should be visited exactly once");
            visited[city] = true;
        }
        assert!(visited.iter().all(|&v| v), "All cities should be visited");
    }

    #[test]
    fn uses_seeded_runs_deterministically() {
        let tsp_data = "
        NAME : custom_small
        TYPE : TSP
        DIMENSION : 6
        EDGE_WEIGHT_TYPE : EUC_2D
        NODE_COORD_SECTION
        1 0 0
        2 1 0
        3 2 0
        4 2 1
        5 1 1
        6 0 1
        EOF
        ";
        let tsp = TspBuilder::parse_str(tsp_data).unwrap();
        let mut lhs = SA2Opt::new_with_seed(tsp.clone(), 9);
        let mut rhs = SA2Opt::new_with_seed(tsp, 9);

        assert_eq!(lhs.seed(), 9);
        assert_eq!(lhs.solve(), rhs.solve());
    }
}
