use crate::{Solution, TspSolver};
use crate::Tsp;

/// The `BranchAndBound` struct implements the branch-and-bound algorithm for solving
/// the Traveling Salesman Problem (TSP). It explores all possible tours and prunes 
/// branches that cannot lead to better solutions than the current best tour.
///
/// # Fields
///
/// - `tsp`: A reference to the `Tsp` instance representing the problem.
/// - `best_tour`: A vector holding the best tour found.
/// - `best_cost`: The total cost of the best tour found.
///
/// # Example
///
/// ```
/// # use ibn_battuta::BranchAndBound;
/// # use ibn_battuta::TspSolver;
/// # use ibn_battuta::TspBuilder;
/// let data = "
/// NAME : example
/// COMMENT : simple example
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
/// let mut solver = BranchAndBound::new(&tsp);
/// let solution = solver.solve();
/// assert_eq!(solution.tour.len(), 5);
/// assert!((solution.length - 13.646824151749852).abs() < f64::EPSILON);
/// ```
pub struct BranchAndBound<'a> {
    tsp: &'a Tsp,
    best_tour: Vec<usize>,
    best_cost: f64,
}

impl<'a> BranchAndBound<'a> {
    /// Creates a new instance of `BranchAndBound` for a given TSP instance.
    ///
    /// # Arguments
    ///
    /// * `tsp` - A reference to a `Tsp` instance that defines the problem to solve.
    ///
    /// # Returns
    ///
    /// A `BranchAndBound` instance initialized for the given TSP.
    ///
    /// # Example
    ///
    /// ```
    /// # use ibn_battuta::BranchAndBound;
    /// # use ibn_battuta::TspBuilder;
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
    /// let tsp = TspBuilder::parse_str(data).unwrap();
    /// let solver = BranchAndBound::new(&tsp);
    /// ```
    pub fn new(tsp: &'a Tsp) -> Self {
        BranchAndBound {
            tsp,
            best_tour: vec![],
            best_cost: f64::INFINITY,
        }
    }

    /// The core recursive function implementing the branch-and-bound search.
    ///
    /// This function explores all possible tours starting from the current tour, pruning
    /// branches when it is clear that further exploration cannot lead to a better solution.
    ///
    /// # Arguments
    ///
    /// * `current_tour` - The current tour of cities being explored.
    /// * `current_cost` - The total cost of the current tour.
    /// * `visited` - A boolean vector indicating which cities have already been visited.
    fn branch_and_bound(&mut self, current_tour: Vec<usize>, current_cost: f64, visited: Vec<bool>) {
        // Base case: if all cities have been visited, check if this is the best tour
        if current_tour.len() == self.tsp.dim() {
            if current_cost < self.best_cost {
                self.best_tour = current_tour.clone();
                self.best_cost = current_cost;
            }
            return;
        }

        // Explore all possible next cities
        let current_node = current_tour.last().unwrap();
        for next_node in 0..self.tsp.dim() {
            if visited[next_node] {
                continue;
            }

            // Calculate the new cost by adding the distance to the next city
            let new_cost = current_cost + self.tsp.weight(*current_node, next_node);
            if new_cost >= self.best_cost {
                continue;  // Prune this branch if the cost exceeds the best known cost
            }

            // Mark the next city as visited and recurse
            let mut new_visited = visited.clone();
            new_visited[next_node] = true;
            let mut new_tour = current_tour.clone();
            new_tour.push(next_node);

            self.branch_and_bound(new_tour, new_cost, new_visited);
        }
    }

    /// Runs the branch-and-bound algorithm to find the optimal tour.
    ///
    /// This function initializes the required data structures and starts the
    /// recursive search for the best tour.
    pub fn run(&mut self) {
        let mut visited = vec![false; self.tsp.dim()];
        visited[0] = true;  // Start from the first city
        self.branch_and_bound(vec![0], 0.0, visited);
    }
}

impl TspSolver for BranchAndBound<'_> {
    /// Solves the TSP using the branch-and-bound algorithm and returns the optimal solution.
    ///
    /// # Example
    ///
    /// ```
    /// # use ibn_battuta::BranchAndBound;
    /// # use ibn_battuta::TspSolver;
    /// # use ibn_battuta::TspBuilder;
    /// let data = "
    /// NAME : example
    /// COMMENT : simple example
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
    /// let mut solver = BranchAndBound::new(&tsp);
    /// let solution = solver.solve();
    /// assert_eq!(solution.tour.len(), 5);
    /// assert!((solution.length - 13.646824151749852).abs() < f64::EPSILON);
    /// ```
    fn solve(&mut self) -> Solution {
        self.run();

        Solution::new(self.best_tour.iter().map(|&i| i as usize).collect(), self.calculate_tour_cost(&self.best_tour))
    }

    /// Returns the best tour found by the branch-and-bound solver.
    ///
    /// # Example
    ///
    /// ```
    /// let data = "
    /// NAME : example
    /// COMMENT : simple example
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
    /// # use ibn_battuta::{BranchAndBound, TspBuilder};
    /// # use ibn_battuta::TspSolver;
    /// let mut solver = BranchAndBound::new(&tsp);
    /// solver.solve();
    /// let tour = solver.tour();
    /// assert_eq!(tour.len(), 5);
    /// ```
    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    /// Returns the cost of traveling between two cities in the TSP instance.
    ///
    /// # Arguments
    ///
    /// * `from` - The starting city.
    /// * `to` - The destination city.
    ///
    /// # Returns
    ///
    /// The cost of traveling between `from` and `to`.
    ///
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}

#[cfg(test)]
mod tests {
    use crate::{BranchAndBound, TspBuilder, TspSolver};

    #[test]
    fn solves_simple_tsp_with_branch_and_bound() {
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
        let mut solver = BranchAndBound::new(&tsp);
        let solution = solver.solve();

        println!("{:?}", solution);
        println!("{:?}", solution.tour);

        assert_eq!(solution.tour.len(), 4);
        assert!((solution.length - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_single_node_with_branch_and_bound() {
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
        let mut solver = BranchAndBound::new(&tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 1);
        assert!((solution.length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_two_nodes_with_branch_and_bound() {
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
        let mut solver = BranchAndBound::new(&tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 2);
        assert!((solution.length - 2.0).abs() < f64::EPSILON);
    }
}
