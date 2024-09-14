use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;

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
/// assert!((solution.length - 4.0).abs() < std::f64::EPSILON);
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
        let mut tour = vec![0];  // Start the tour from city 0
        self.solve_recursive(&mut tour, 0.0);  // Recursively find the best tour

        Solution::new(
            self.best_tour.iter().map(|&i| i as usize).collect(),
            self.best_cost,
        )
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
            best_cost: f64::INFINITY,  // Initialize with infinity cost
        }
    }

    /// Recursively explores all possible tours and updates the best one found.
    ///
    /// # Arguments
    /// * `tour` - The current partial tour being explored.
    /// * `cost` - The current cost of the partial tour.
    fn solve_recursive(&mut self, tour: &mut Vec<usize>, cost: f64) {
        if tour.len() == self.tsp.dim() {
            // If we've visited all cities, complete the tour by returning to the starting city
            let last = tour.last().unwrap();
            let cost = cost + self.tsp.weight(*last, tour[0]);

            // Update the best tour and cost if this tour is better
            if cost < self.best_cost {
                self.best_cost = cost;
                self.best_tour = tour.clone();
            }
        } else {
            // Explore all cities that haven't been visited yet
            for i in 0..self.tsp.dim() {
                if !tour.contains(&i) {
                    // Add the next city to the tour and calculate the new cost
                    let mut new_tour = tour.clone();
                    new_tour.push(i);
                    let new_cost = cost + self.tsp.weight(*tour.last().unwrap(), i);

                    // Recursively solve for the new tour
                    self.solve_recursive(&mut new_tour, new_cost);
                }
            }
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