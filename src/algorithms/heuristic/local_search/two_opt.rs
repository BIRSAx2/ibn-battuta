use crate::algorithms::{Solution, TspSolver};
use tspf::Tsp;

pub struct TwoOpt<'a> {
    tsp: &'a Tsp,
    tour: Vec<usize>,
    cost: f64,
    verbose: bool,
    base_tour: Vec<usize>,
}


impl<'a> TwoOpt<'a> {
    pub fn new(tsp: &Tsp) -> TwoOpt {
        TwoOpt {
            tsp,
            tour: vec![],
            cost: 0.0,
            verbose: false,
            base_tour: tsp.node_coords().iter().map(|node| *node.0).collect(),
        }
    }

    pub fn from(tsp: &Tsp, base_tour: Vec<usize>, verbose: bool) -> TwoOpt {
        TwoOpt {
            tsp,
            tour: vec![],
            cost: 0.0,
            verbose,
            base_tour,
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
    fn solve(&mut self) -> Solution {
        self.tour = self.base_tour.clone();
        self.cost = self.calculate_tour_cost();

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
    use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
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

        let mut nn = NearestNeighbor::new(&tsp);
        let base_tour = nn.solve().tour;
        let mut solver = TwoOpt::from(&tsp, base_tour, false);
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
        let mut solver = TwoOpt::from(&tsp, base_tour, false);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}