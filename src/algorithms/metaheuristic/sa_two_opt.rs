use crate::{SimulatedAnnealing, Solution, Tsp, TspSolver, TwoOpt};

pub struct SA2Opt {
    tsp: Tsp,
    sa: SimulatedAnnealing,
}

impl SA2Opt {
    pub fn new(tsp: Tsp) -> SA2Opt {
        let acs = SimulatedAnnealing::new(tsp.clone());

        SA2Opt {
            tsp,
            sa: acs,
        }
    }
}
impl TspSolver for SA2Opt {
    fn solve(&mut self) -> Solution {
        let base_solution = self.sa.solve();
        TwoOpt::from(self.tsp.clone(), base_solution.tour, false).solve()
    }

    fn tour(&self) -> Vec<usize> {
        todo!("Implement tour method for SA2Opt")
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        format!("SA2Opt")
    }
}
#[cfg(test)]
mod tests {
    use crate::algorithms::TspSolver;
    use crate::sa_two_opt::SA2Opt;
    use crate::{Tsp, TspBuilder};

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        test_instance(tsp);
    }

    fn test_instance(tsp: Tsp) {
        let size = tsp.dim();

        // let n = tsp.dim();
        let mut solver = SA2Opt::new(tsp);

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