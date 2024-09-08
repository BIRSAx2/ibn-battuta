use crate::{AntColonySystem, Solution, Tsp, TspSolver, TwoOpt};

pub struct ACS2Opt {
    tsp: Tsp,
    acs: AntColonySystem,
}

impl ACS2Opt {
    pub fn with_options(tsp: Tsp, alpha: f64, beta: f64, rho: f64, q0: f64, num_ants: usize, max_iterations: usize, candidate_list_size: usize) -> ACS2Opt {
        let acs = AntColonySystem::with_options(tsp.clone(), alpha, beta, rho, q0, num_ants, max_iterations, candidate_list_size);

        ACS2Opt {
            tsp,
            acs: acs,
        }
    }
}
impl TspSolver for ACS2Opt {
    fn solve(&mut self) -> Solution {
        let base_solution = self.acs.solve();
        TwoOpt::from(self.tsp.clone(), base_solution.tour, false).solve()
    }

    fn tour(&self) -> Vec<usize> {
        todo!("Implement tour method for ACS2Opt")
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        format!("ACS2Opt")
    }
}
#[cfg(test)]
mod tests {
    use crate::algorithms::TspSolver;
    use crate::ant_colony::acs_two_opt::ACS2Opt;
    use crate::{Tsp, TspBuilder};

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
        let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 20, 1000, 15);


        let solution = solver.solve();

        println!("{:?}", solution);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 20, 1000, 15);
        let solution = solver.solve();

        println!("{:?}", solution);
    }

    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();

        // let n = tsp.dim();
        let mut solver = ACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.9, 20, 1000, 15);

        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }

    #[test]
    fn test_gr666() {
        let path = "data/tsplib/st70.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        test_instance(tsp);
    }

    #[test]
    fn test_p43() {
        let path = "data/tsplib/berlin52.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        test_instance(tsp);
    }
}