use crate::algorithms::utils::SolverConfig;
use crate::algorithms::{Solution, TspSolver};
use tspf::Tsp;


pub struct BruteForce<'a> {
    tsp: &'a Tsp,
    best_tour: Vec<usize>,
    best_cost: f64,
}


impl TspSolver for BruteForce<'_> {
    fn solve(&mut self, _options: &SolverConfig) -> Solution {
        let mut tour = vec![0];
        self.solve_recursive(&mut tour, 0.0);

        Solution::new(self.best_tour.iter().map(|&i| i as usize).collect(), self.best_cost)
    }
    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}


impl<'a> BruteForce<'a> {
    pub fn new(tsp: &'a Tsp) -> Self {
        BruteForce {
            tsp,
            best_tour: vec![],
            best_cost: f64::INFINITY,
        }
    }

    fn solve_recursive(&mut self, tour: &mut Vec<usize>, cost: f64) {
        if tour.len() == self.tsp.dim() {
            let last = tour.last().unwrap();
            let cost = cost + self.tsp.weight(*last, tour[0]);
            if cost < self.best_cost {
                self.best_cost = cost;
                self.best_tour = tour.clone();
            }
        } else {
            for i in 0..self.tsp.dim() {
                if !tour.contains(&i) {
                    let mut new_tour = tour.clone();
                    new_tour.push(i);
                    let new_cost = cost + self.tsp.weight(*tour.last().unwrap(), i);
                    self.solve_recursive(&mut new_tour, new_cost);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::exact::brute_force::BruteForce;
    use crate::algorithms::utils::SolverConfig;
    use crate::algorithms::TspSolver;
    use tspf::TspBuilder;

    #[test]
    fn test() {
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

        let mut solver = BruteForce::new(&tsp);
        let options = SolverConfig::new_brute_force(1000);

        let solution = solver.solve(&options);

        assert_eq!(solution.tour.len(), 5);
        assert!((solution.total - 13.646824151749852).abs() < f64::EPSILON);
    }
}