use tspf::Tsp;

use crate::algorithms::{SolverOptions, TspSolver, Solution};

pub struct ThreeOpt<'a> {
    tsp: &'a Tsp,
    tour: Vec<usize>,
    cost: f64,
    options: SolverOptions,
}

impl ThreeOpt<'_> {
    pub fn new(tsp: &Tsp, options: SolverOptions) -> ThreeOpt {
        ThreeOpt {
            tsp,
            tour: vec![],
            cost: 0.0,
            options,
        }
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
                for j in i + 2..n - 1 {
                    for k in j + 2..n + (i > 0) as usize {
                        if k >= n { continue; } // Ensure k stays in bounds

                        let new_tours = self.generate_new_tours(i, j, k);

                        for new_tour in new_tours {
                            let new_cost = self.calculate_tour_cost_with(&new_tour);

                            if new_cost < self.cost {
                                self.tour = new_tour;
                                self.cost = new_cost;
                                improved = true;

                                if self.options.verbose {
                                    println!(
                                        "3OPT: Improved tour at segments ({}, {}, {}), new cost: {}",
                                        i, j, k, self.cost
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn generate_new_tours(&self, i: usize, j: usize, k: usize) -> Vec<Vec<usize>> {
        let mut new_tours = Vec::new();
        let n = self.tour.len();

        let mut tour1 = self.tour.clone();
        let mut tour2 = self.tour.clone();
        let mut tour3 = self.tour.clone();
        let mut tour4 = self.tour.clone();
        let mut tour5 = self.tour.clone();
        let mut tour6 = self.tour.clone();
        let mut tour7 = self.tour.clone();

        // Case 1: no change (do nothing)

        // Case 2: 2-opt between i+1 and j (reverse segment between i+1 and j)
        tour1[i + 1..=j].reverse();

        // Case 3: 2-opt between j+1 and k (reverse segment between j+1 and k)
        tour2[j + 1..=k].reverse();

        // Case 4: 2-opt between i+1 and k (reverse segment between i+1 and k)
        tour3[i + 1..=k].reverse();

        // Case 5: 3-opt with reversing segments i+1 to j and j+1 to k
        tour4[i + 1..=j].reverse();
        tour4[j + 1..=k].reverse();

        // Case 6: 3-opt with reversing segments i+1 to j and i+1 to k
        tour5[i + 1..=j].reverse();
        tour5[i + 1..=k].reverse();

        // Case 7: 3-opt with reversing segments j+1 to k and i+1 to k
        tour6[j + 1..=k].reverse();
        tour6[i + 1..=k].reverse();

        // Case 8: 3-opt with reversing all segments
        tour7[i + 1..=j].reverse();
        tour7[j + 1..=k].reverse();
        tour7[i + 1..=k].reverse();

        new_tours.push(tour1);
        new_tours.push(tour2);
        new_tours.push(tour3);
        new_tours.push(tour4);
        new_tours.push(tour5);
        new_tours.push(tour6);
        new_tours.push(tour7);

        new_tours
    }

    fn calculate_tour_cost_with(&self, tour: &Vec<usize>) -> f64 {
        let mut cost = 0.0;
        let len = tour.len();
        for i in 0..len {
            let from = tour[i];
            let to = tour[(i + 1) % len];
            cost += self.tsp.weight(from, to) as f64;
        }
        cost
    }
}

impl TspSolver for ThreeOpt<'_> {
    fn solve(&mut self, options: &SolverOptions) -> Solution {
        let mut base_solver = options.base_solver.create(&self.tsp);
        let base_solution = base_solver.solve(options);
        self.tour = base_solution.tour.clone();
        self.cost = base_solution.total;

        if options.verbose {
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
    use tspf::TspBuilder;
    use crate::algorithms::{SolverOptions, TspSolver};
    use super::*;

    #[test]
    fn test_three_opt() {
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

        let mut options = SolverOptions::default();
        options.verbose = true;
        let mut solver = ThreeOpt::new(&tsp, options);
        let solution = solver.solve(&SolverOptions::default());

        println!("{:?}", solution);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();
        let options = SolverOptions::default();
        let mut solver = ThreeOpt::new(&tsp, options);
        let solution = solver.solve(&SolverOptions::default());
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}
