use crate::algorithms::{Solution, TspSolver};
use std::f64;
use crate::Tsp;
pub struct Greedy {
    tsp: Box<Tsp>,
    visited: Vec<bool>,
    tour: Vec<usize>,
    cost: f64,
}


impl Greedy {
    pub fn new(tsp: Box<Tsp>) -> Self {
        let n = tsp.dim();
        Greedy {
            tsp,
            visited: vec![false; n],
            tour: Vec::with_capacity(n),
            cost: 0.0,
        }
    }
}

impl TspSolver for Greedy {
    fn solve(&mut self) -> Solution {
        let n = self.tsp.dim();
        let dist = |i: usize, j: usize| self.tsp.weight(i, j);

        self.visited[0] = true;
        self.tour.push(0);

        for _ in 1..n {
            let last = *self.tour.last().unwrap();
            let mut best = (0, f64::INFINITY);
            for i in 0..n {
                if !self.visited[i] {
                    let cost = dist(last, i);
                    if cost < best.1 {
                        best = (i, cost);
                    }
                }
            }
            self.visited[best.0] = true;
            self.tour.push(best.0);
            self.cost += best.1;
        }

        self.cost += dist(*self.tour.last().unwrap(), 0);

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
    use crate::algorithms::TspSolver;
    use tspf::TspBuilder;

    #[test]
    fn test_greedy() {
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

        let mut solver = Greedy::new(Box::new(tsp));
        let solution = solver.solve();

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();

        let mut solver = Greedy::new(Box::new(tsp));

        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}