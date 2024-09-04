pub mod exact;
pub mod heuristic;
pub mod metaheuristic;
pub mod utils;

pub use exact::*;
pub use heuristic::*;
pub use metaheuristic::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    pub tour: Vec<usize>,
    pub total: f64,
}

impl Default for Solution {
    fn default() -> Self {
        Solution {
            tour: vec![],
            total: 0.0,
        }
    }
}

impl Solution {
    pub fn new(tour: Vec<usize>, total: f64) -> Self {
        Solution {
            tour,
            total,
        }
    }
}

pub trait TspSolver {
    fn solve(&mut self) -> Solution;
    fn tour(&self) -> Vec<usize>;
    fn cost(&self, from: usize, to: usize) -> f64;
    fn calculate_tour_cost(&self, tour: &Vec<usize>) -> f64 {
        let mut total_cost = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total_cost += self.cost(from, to);
        }
        total_cost
    }

    fn format_name(&self) -> String {
        format!("{}", "TspSolver")
    }
}

impl std::fmt::Display for dyn TspSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.format_name())
    }
}
