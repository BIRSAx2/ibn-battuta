use crate::algorithms::utils::SolverConfig;

mod exact;
mod heuristic;
mod metaheuristic;
mod utils;


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
    fn solve(&mut self, options: &SolverConfig) -> Solution;
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
}