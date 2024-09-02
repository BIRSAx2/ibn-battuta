use crate::algorithms::utils::{HeuristicAlgorithmConfig, SolverConfig};
use crate::algorithms::{Solution, TspSolver};
use tspf::Tsp;

pub struct TwoOpt<'a> {
    tsp: &'a Tsp,
    tour: Vec<usize>,
    cost: f64,
    verbose: bool,
}

impl TwoOpt<'_> {
    pub fn new(tsp: &Tsp) -> TwoOpt {
        TwoOpt {
            tsp,
            tour: vec![],
            cost: 0.0,
            verbose: false,
        }
    }

    fn swap_2opt(tour: &mut Vec<usize>, i: usize, k: usize) {
        tour[i..=k].reverse();
    }

    fn calculate_tour_cost(&self) -> f64 {
        let mut cost = 0.0;
        let len = self.tour.len();
        for i in 0..len {
            let from = self.tour[i];
            let to = self.tour[(i + 1) % len];
            cost += self.tsp.weight(from, to) as f64;
        }
        cost
    }

    pub fn optimize(&mut self) {
        let n = self.tour.len();
        let mut improved = true;

        while improved {
            improved = false;
            for i in 0..n - 2 {
                for j in i + 2..n {
                    let current_distance = self.tsp.weight(self.tour[i], self.tour[i + 1]) as f64
                        + self.tsp.weight(self.tour[j], self.tour[(j + 1) % n]) as f64;

                    let new_distance = self.tsp.weight(self.tour[i], self.tour[j]) as f64
                        + self.tsp.weight(self.tour[i + 1], self.tour[(j + 1) % n]) as f64;

                    if new_distance < current_distance {
                        Self::swap_2opt(&mut self.tour, i + 1, j);
                        self.cost = self.calculate_tour_cost();
                        improved = true;

                        if self.verbose {
                            println!(
                                "2OPT: Swapped edges ({} - {}) and ({} - {}), new cost: {}",
                                self.tour[i],
                                self.tour[i + 1],
                                self.tour[j],
                                self.tour[(j + 1) % n],
                                self.cost
                            );
                        }
                    }
                }
            }
        }
    }
}

impl TspSolver for TwoOpt<'_> {
    fn solve(&mut self, options: &SolverConfig) -> Solution {
        let (base_solver, verbose) = match options {
            SolverConfig::HeuristicAlgorithm(HeuristicAlgorithmConfig::LocalSearch { base_solver, verbose, .. }) => {
                (base_solver, verbose)
            }
            _ => panic!("Invalid solver configuration"),
        };

        self.verbose = *verbose;

        let mut base_solver = base_solver.create(&self.tsp, options);
        let base_solution = base_solver.solve(options);
        self.tour = base_solution.tour.clone();
        self.cost = base_solution.total;

        if self.verbose {
            println!("Initial tour: {:?}", self.tour);
            println!("Initial cost: {}", self.cost);
        }

        self.optimize();

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
    use crate::algorithms::heuristic::local_search::two_opt::TwoOpt;
    use crate::algorithms::utils::{Solver, SolverConfig};
    use crate::algorithms::TspSolver;
    use tspf::TspBuilder;

    #[test]
    fn test_two_opt() {
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

        let options = SolverConfig::default();
        let mut solver = TwoOpt::new(&tsp);
        let solution = solver.solve(&options);

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let options = SolverConfig::new_local_search(Solver::NearestNeighbor, false, 1000);
        let mut solver = TwoOpt::new(&tsp);
        let solution = solver.solve(&options);
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}