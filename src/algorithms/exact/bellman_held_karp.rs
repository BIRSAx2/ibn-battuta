use crate::algorithms::{Solution, TspSolver};
use std::f64;
use crate::Tsp;

/// The `BellmanHeldKarp` struct implements the Held-Karp dynamic programming algorithm 
/// for solving the Traveling Salesman Problem (TSP). It stores the TSP instance, optimal 
/// subproblem results, and the best tour and cost.
///
/// # Fields
///
/// - `tsp`: The TSP instance to solve.
/// - `opt`: A 2D table storing the cost of visiting a subset of nodes, ending at a particular node.
/// - `best_tour`: The optimal tour that visits every node once and returns to the start.
/// - `best_cost`: The total cost of the optimal tour.
///
/// # Example
///
/// ```
/// # use ibn_battuta::algorithms::BellmanHeldKarp;
/// # use ibn_battuta::algorithms::TspSolver;
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
/// let mut solver = BellmanHeldKarp::new(tsp);
/// let solution = solver.solve();
/// assert_eq!(solution.tour.len(), 4);
/// assert!((solution.length - 4.0).abs() < f64::EPSILON);
/// ```
pub struct BellmanHeldKarp {
    tsp: Tsp,
    opt: Vec<Vec<Option<f64>>>,
    best_tour: Vec<usize>,
    best_cost: f64,
}

impl BellmanHeldKarp {
    /// Creates a new instance of `BellmanHeldKarp` for a given TSP instance.
    ///
    /// # Arguments
    ///
    /// * `tsp` - A `Tsp` instance that defines the problem to solve.
    ///
    /// # Returns
    ///
    /// A `BellmanHeldKarp` instance initialized for the given TSP.
    ///
    /// # Example
    ///
    /// ```
    /// # use ibn_battuta::BellmanHeldKarp;
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
    /// let solver = BellmanHeldKarp::new(tsp);
    /// ```
    pub fn new(tsp: Tsp) -> Self {
        let n = tsp.dim();
        let opt = vec![vec![None; 1 << (n - 1)]; n - 1];
        BellmanHeldKarp {
            tsp,
            opt,
            best_tour: Vec::with_capacity(n),
            best_cost: f64::INFINITY,
        }
    }

    /// Solves the TSP using the Bellman-Held-Karp dynamic programming algorithm.
    ///
    /// This function computes the shortest possible tour that visits every city 
    /// once and returns to the starting point. The result is stored in the 
    /// `best_tour` and `best_cost` fields.
    ///
    /// # Example
    ///
    /// ```
    /// # use ibn_battuta::BellmanHeldKarp;
    /// # use ibn_battuta::TspSolver;
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
    /// let mut solver = BellmanHeldKarp::new(tsp);
    /// solver.bellman_held_karp();
    /// ```
    pub fn bellman_held_karp(&mut self) {
        let n = self.tsp.dim();
        let dist = |i: usize, j: usize| self.tsp.weight(i, j);

        // Initialize the base cases
        for i in 0..n - 1 {
            self.opt[i][1 << i] = Some(dist(i, n - 1));
        }

        // Iterate over subsets of increasing size
        for size in 2..n {
            for s in 1..(1 << (n - 1)) {
                if (s as i32).count_ones() as usize == size {
                    for t in 0..n - 1 {
                        if (s & (1 << t)) != 0 {
                            let mut min_cost = f64::INFINITY;
                            let prev_s = s & !(1 << t);
                            for q in 0..n - 1 {
                                if (prev_s & (1 << q)) != 0 {
                                    if let Some(cost) = self.opt[q][prev_s] {
                                        min_cost = f64::min(min_cost, cost + dist(q, t));
                                    }
                                }
                            }
                            self.opt[t][s] = Some(min_cost);
                        }
                    }
                }
            }
        }

        // Calculate the minimum cost to complete the tour
        for t in 0..n - 1 {
            if let Some(cost) = self.opt[t][(1 << (n - 1)) - 1] {
                let final_cost = cost + dist(t, n - 1);
                if final_cost < self.best_cost {
                    self.best_cost = final_cost;
                    self.best_tour = self.build_tour(t, (1 << (n - 1)) - 1);
                }
            }
        }
    }

    /// Constructs the optimal tour using the previously computed results.
    ///
    /// # Arguments
    ///
    /// * `last` - The last city visited in the optimal tour.
    /// * `s` - The bitmask representing the set of cities visited.
    ///
    /// # Returns
    ///
    /// A vector representing the optimal tour.
    fn build_tour(&self, mut last: usize, mut s: usize) -> Vec<usize> {
        let mut tour = vec![last];
        for _ in 1..self.tsp.dim() - 1 {
            for i in 0..self.tsp.dim() - 1 {
                if s & (1 << i) != 0 {
                    if let Some(cost) = self.opt[i][s & !(1 << last)] {
                        if cost + self.tsp.weight(i, last) == self.opt[last][s].unwrap() {
                            tour.push(i);
                            s &= !(1 << last);
                            last = i;
                            break;
                        }
                    }
                }
            }
        }
        tour.push(self.tsp.dim() - 1);
        tour.reverse();
        tour
    }
}

impl TspSolver for BellmanHeldKarp {
    /// Solves the TSP and returns the optimal solution.
    ///
    /// # Example
    ///
    /// ```
    /// # use ibn_battuta::BellmanHeldKarp;
    /// # use ibn_battuta::TspSolver;
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
    /// let mut solver = BellmanHeldKarp::new(tsp);
    /// let solution = solver.solve();
    /// assert_eq!(solution.tour.len(), 4);
    /// assert!((solution.length - 4.0).abs() < f64::EPSILON);
    /// ```
    fn solve(&mut self) -> Solution {
        if self.tsp.dim() == 1 {
            return Solution::new(vec![0], 0.0);
        }
        self.bellman_held_karp();
        Solution::new(self.best_tour.iter().map(|&i| i).collect(), self.best_cost)
    }

    /// Returns the best tour found by the solver.
    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    /// Returns the cost of traveling between two cities.
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}


#[cfg(test)]
mod tests {
    use crate::BellmanHeldKarp;
    use crate::TspSolver;
    use crate::TspBuilder;

    #[test]
    fn solves_simple_tsp() {
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
        let mut solver = BellmanHeldKarp::new(tsp);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert!((solution.length - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_single_node() {
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
        let mut solver = BellmanHeldKarp::new(tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 1);
        assert!((solution.length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_two_nodes() {
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
        let mut solver = BellmanHeldKarp::new(tsp);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 2);
        assert!((solution.length - 2.0).abs() < f64::EPSILON);
    }
}