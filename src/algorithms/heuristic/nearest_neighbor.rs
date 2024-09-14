use crate::algorithms::{Solution, TspSolver};
use crate::Tsp;
use std::f64;

/// The Nearest Neighbor (NN) heuristic for solving the Traveling Salesman Problem (TSP).
///
/// This algorithm starts at an arbitrary city (city 0 in this case), and iteratively visits the closest unvisited city.
/// It continues until all cities have been visited, and then returns to the starting city to complete the tour.
///
/// # Attributes
/// * `tsp`: The TSP instance.
/// * `visited`: A boolean vector indicating whether each city has been visited.
/// * `tour`: The tour of cities visited.
/// * `cost`: The total cost (distance) of the tour.
///
/// # Example
/// ```
/// use ibn_battuta::TspSolver;
/// use ibn_battuta::TspBuilder;
/// use ibn_battuta::NearestNeighbor;
///
/// let data = "
/// NAME : example
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
/// let mut solver = NearestNeighbor::new(tsp);
/// let solution = solver.solve();
///
/// assert_eq!(solution.tour.len(), 4);
/// assert!((solution.length - 3.0).abs() < std::f64::EPSILON);
/// ```
pub struct NearestNeighbor {
    tsp: Tsp,
    visited: Vec<bool>,
    tour: Vec<usize>,
    cost: f64,
}

impl NearestNeighbor {
    /// Creates a new `NearestNeighbor` solver for the given TSP instance.
    ///
    /// # Arguments
    /// * `tsp` - The `Tsp` problem instance to be solved.
    ///
    /// Initializes the solver with the required attributes such as the `visited` vector
    /// and `tour` vector with the appropriate size based on the number of cities.
    pub fn new(tsp: Tsp) -> Self {
        let n = tsp.dim();
        NearestNeighbor {
            tsp,
            visited: vec![false; n],
            tour: Vec::with_capacity(n),
            cost: 0.0,
        }
    }
}

impl TspSolver for NearestNeighbor {
    /// Solves the TSP using the nearest neighbor heuristic.
    ///
    /// Starts from city 0, then iteratively selects the nearest unvisited city
    /// until all cities are visited. The total cost of the tour is calculated
    /// as the sum of distances between consecutive cities in the tour.
    ///
    /// Returns the solution which includes the tour and its total cost.
    fn solve(&mut self) -> Solution {
        let n = self.tsp.dim();
        let mut current_city = 0; // Start at city 0
        self.visited[current_city] = true;
        self.tour.push(current_city);

        // Visit all cities
        for _ in 1..n {
            let mut next_city = None;
            let mut min_distance = f64::MAX;

            // Find the nearest unvisited city
            for city in 0..n {
                if !self.visited[city] {
                    let distance = self.tsp.weight(current_city, city);
                    if distance < min_distance {
                        min_distance = distance;
                        next_city = Some(city);
                    }
                }
            }

            // Move to the nearest unvisited city
            if let Some(city) = next_city {
                self.visited[city] = true;
                self.tour.push(city);
                self.cost += min_distance;
                current_city = city;
            }
        }

        // Complete the tour by returning to the starting city
        self.cost += self.tsp.weight(current_city, self.tour[0]);

        Solution {
            tour: self.tour.clone(),
            length: self.cost,
        }
    }

    /// Returns the computed tour after solving the problem.
    fn tour(&self) -> Vec<usize> {
        self.tour.clone()
    }

    /// Returns the cost (distance) between two cities.
    ///
    /// # Arguments
    /// * `from` - The starting city.
    /// * `to` - The destination city.
    ///
    /// Returns the edge weight (cost) between the two cities.
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    /// Returns the format name of the algorithm ("NN").
    fn format_name(&self) -> String {
        "NN".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::TspSolver;
    use crate::TspBuilder;

    #[test]
    fn nearest_neighbor_handles_single_city() {
        let data = "
        NAME : single_city
        TYPE : TSP
        DIMENSION : 1
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 0.0 0.0
        EOF
";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = NearestNeighbor::new(tsp);
        let solution = solver.solve();
        assert_eq!(solution.tour.len(), 1);
        assert_eq!(solution.length, 0.0);
    }

    #[test]
    fn nearest_neighbor_handles_two_cities() {
        let data = "
        NAME : two_cities
        TYPE : TSP
        DIMENSION : 2
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 0.0 0.0
          2 1.0 0.0
        EOF
        ";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = NearestNeighbor::new(tsp);
        let solution = solver.solve();
        assert_eq!(solution.tour.len(), 2);
        assert!((solution.length - 2.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn nearest_neighbor_handles_non_euclidean_distances() {
        let data = "
        NAME : non_euclidean
        TYPE : TSP
        DIMENSION : 3
        EDGE_WEIGHT_TYPE: EXPLICIT
        EDGE_WEIGHT_FORMAT: FULL_MATRIX
        EDGE_WEIGHT_SECTION
          0 2 9
          2 0 6
          9 6 0
        EOF
        ";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = NearestNeighbor::new(tsp);
        let solution = solver.solve();
        assert_eq!(solution.tour.len(), 3);
        approx::assert_relative_eq!(solution.length, 17.0);
    }
}
