use crate::{RedBlackACS, Solution, Tsp, TspSolver, TwoOpt};

pub struct RBACS2Opt {
    tsp: Tsp,
    rbacs: RedBlackACS,
}

impl RBACS2Opt {
    pub fn with_options(tsp: Tsp, alpha: f64, beta: f64, rho_red: f64, rho_black: f64, q0: f64,
                        num_ants: usize, max_iterations: usize, candidate_list_size: usize) -> RBACS2Opt {
        let acs = RedBlackACS::new(tsp.clone(), alpha, beta, rho_red, rho_black, q0,
                                   num_ants, max_iterations, candidate_list_size);

        RBACS2Opt {
            tsp,
            rbacs: acs,
        }
    }
}
impl TspSolver for RBACS2Opt {
    fn solve(&mut self) -> Solution {
        let base_solution = self.rbacs.solve();
        TwoOpt::from(self.tsp.clone(), base_solution.tour, false).solve()
    }

    fn tour(&self) -> Vec<usize> {
        todo!("Implement tour method for ACS2Opt")
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        format!("RBACS2Opt")
    }
}
#[cfg(test)]
mod tests {
    use crate::algorithms::TspSolver;
    use crate::ant_colony::rbacs_two_opt::RBACS2Opt;
    use crate::{Tsp, TspBuilder};

    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();

        // let n = tsp.dim();
        let mut solver = RBACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.2, 0.9, 10, 1000, 15);


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