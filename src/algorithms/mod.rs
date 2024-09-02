use std::hash::RandomState;
use tspf::{Tsp};
use crate::algorithms::exact::branch_and_bound::BranchAndBound;
use crate::algorithms::exact::brute_force::BruteForce;
use crate::algorithms::heuristic::local_search::two_opt::TwoOpt;
use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;

mod exact;
mod heuristic;
mod metaheuristic;

#[derive(Clone, Debug, PartialEq)]
pub enum Solvers {
    BruteForce,
    BranchAndBound,
    NearestNeighbor,
    TwoOpt,
    // ThreeOpt,
    Greedy,
}

impl Solvers {
    pub fn create<'a>(&self, tsp: &'a Tsp) -> Box<dyn TspSolver + 'a> {
        match self {
            Solvers::BruteForce => Box::new(BruteForce::new(tsp)),
            Solvers::BranchAndBound => Box::new(BranchAndBound::new(tsp)),
            Solvers::NearestNeighbor => Box::new(NearestNeighbor::new(tsp)),
            Solvers::TwoOpt => Box::new(TwoOpt::new(tsp, SolverOptions::default())),
            // Solvers::ThreeOpt => Box::new(heuristic::local_search::three_opt::ThreeOpt::new(tsp, Box::new(NearestNeighbor::new(&tsp)))),
            Solvers::Greedy => Box::new(heuristic::greedy::Greedy::new(tsp)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolverOptions {
    pub epochs: u32,
    pub show_progress: bool,
    pub verbose: bool,
    pub base_solver: Solvers,
    pub max_iterations: u32,
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
    pub population_size: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub max_generations: usize,
}


impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            epochs: 1,
            show_progress: false,
            verbose: true,
            base_solver: Solvers::NearestNeighbor,
            max_iterations: 1000,
            cooling_rate: 0.095,
            initial_temperature: 10000.0,
            min_temperature: 0.001,
            population_size: 100,
            tournament_size: 5,
            mutation_rate: 0.01,
            max_generations: 1000,
        }
    }
}


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
    fn solve(&mut self, options: &SolverOptions) -> Solution;
    fn tour(&self) -> Vec<usize>;
    fn cost(&self, from: usize, to: usize) -> f64;
    fn calculate_tour_cost(&self) -> f64 {
        let mut cost = 0.0;
        let len = self.tour().len();
        for i in 0..len {
            let from = self.tour()[i];
            let to = self.tour()[(i + 1) % len];
            cost += self.cost(from, to) as f64;
        }
        cost
    }
}