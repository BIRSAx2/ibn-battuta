use crate::algorithms::{Solution, TspSolver};
use std::f64;
use crate::parser::Tsp;
use rand::prelude::*;

pub struct LinKernighan<'a> {
    tsp: &'a Tsp,
    tour: Vec<usize>,
    cost: f64,
    verbose: bool,
    max_iterations: usize,
}

// TODO: Add verbose output
impl<'a> LinKernighan<'a> {
    pub fn new(tsp: &'a Tsp) -> LinKernighan<'a> {
        let mut result = LinKernighan {
            tsp,
            tour: vec![],
            cost: 0.0,
            verbose: false,
            max_iterations: 1000,
        };

        result.initial_tour();

        result
    }

    pub fn with_options(tsp: &'a Tsp, base_tour: Vec<usize>, verbose: bool, max_iterations: usize) -> LinKernighan<'a> {
        LinKernighan {
            tsp,
            tour: base_tour.clone(),
            cost: 0.0,
            verbose,
            max_iterations,
        }
    }

    fn initial_tour(&mut self) {
        let mut rng = rand::thread_rng();
        self.tour = (0..self.tsp.dim()).collect();
        self.tour.shuffle(&mut rng);
        self.cost = self.calculate_tour_cost();
    }

    fn calculate_tour_cost(&self) -> f64 {
        let mut cost = 0.0;
        for i in 0..self.tour.len() {
            let from = self.tour[i];
            let to = self.tour[(i + 1) % self.tour.len()];
            cost += self.cost(from, to);
        }
        cost
    }

    fn improve_tour(&mut self) -> bool {
        for i in 0..self.tour.len() {
            if self.improve_step(i) {
                return true;
            }
        }
        false
    }

    fn improve_step(&mut self, start: usize) -> bool {
        let mut t = vec![start];
        let mut gain = 0.0;

        loop {
            if let Some((next, new_gain)) = self.find_next(&t, gain) {
                t.push(next);
                gain = new_gain;

                if gain > 0.0 && self.is_tour_feasible(&t) {
                    self.apply_move(&t);
                    return true;
                }

                if t.len() >= self.tour.len() - 1 {
                    break;
                }
            } else {
                break;
            }
        }

        false
    }

    fn find_next(&self, t: &Vec<usize>, current_gain: f64) -> Option<(usize, f64)> {
        let mut best_next = None;
        let mut best_gain = current_gain;

        for i in 0..self.tour.len() {
            if !t.contains(&i) {
                let gain = self.calculate_gain(t, i);
                if gain > best_gain {
                    best_gain = gain;
                    best_next = Some(i);
                }
            }
        }

        best_next.map(|next| (next, best_gain))
    }

    fn calculate_gain(&self, t: &Vec<usize>, next: usize) -> f64 {
        let last = t[t.len() - 1];
        let first = t[0];
        let removed_edge = self.cost(last, self.tour[(last + 1) % self.tour.len()]);
        let added_edge = self.cost(last, next);
        let closing_edge = if t.len() == self.tour.len() - 1 {
            self.cost(next, first)
        } else {
            0.0
        };

        removed_edge - added_edge - closing_edge
    }

    fn is_tour_feasible(&self, t: &Vec<usize>) -> bool {
        t.len() == self.tour.len() && t[0] == t[t.len() - 1]
    }

    fn apply_move(&mut self, t: &Vec<usize>) {
        let mut new_tour = vec![0; self.tour.len()];
        for i in 0..t.len() - 1 {
            new_tour[i] = self.tour[t[i]];
        }
        self.tour = new_tour;
        self.cost = self.calculate_tour_cost();
    }
}

impl TspSolver for LinKernighan<'_> {
    fn solve(&mut self) -> Solution {
        let mut iterations = 0;
        while iterations < self.max_iterations {
            if !self.improve_tour() {
                break;
            }
            iterations += 1;
        }

        Solution {
            tour: self.tour.clone(),
            total: self.cost,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
    use crate::algorithms::TspSolver;
    use tspf::TspBuilder;

    #[test]
    fn test_example() {
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
        let mut nn = NearestNeighbor::new(&tsp);
        let base_tour = nn.solve().tour;
        let mut solver = LinKernighan::with_options(&tsp, base_tour, false, 1000);

        let solution = solver.solve();

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let mut nn = NearestNeighbor::new(&tsp);
        let base_tour = nn.solve().tour;
        let mut solver = LinKernighan::with_options(&tsp, base_tour, false, 1000);
        let solution = solver.solve();
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}