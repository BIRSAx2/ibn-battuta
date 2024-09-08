use crate::{GeneticAlgorithm, Solution, Tsp, TspSolver, TwoOpt};

pub struct GA2Opt {
    tsp: Tsp,
    ga: GeneticAlgorithm,
}

impl GA2Opt {
    pub fn with_options(tsp: Tsp, population_size: usize,
                        elite_size: usize,
                        crossover_rate: f64,
                        mutation_rate: f64,
                        max_generations: usize, ) -> GA2Opt {
        let acs = GeneticAlgorithm::with_options(tsp.clone(), population_size,
                                                 elite_size,
                                                 crossover_rate,
                                                 mutation_rate,
                                                 max_generations, );

        GA2Opt {
            tsp,
            ga: acs,
        }
    }
}
impl TspSolver for GA2Opt {
    fn solve(&mut self) -> Solution {
        let base_solution = self.ga.solve();
        TwoOpt::from(self.tsp.clone(), base_solution.tour, false).solve()
    }

    fn tour(&self) -> Vec<usize> {
        todo!("Implement tour method for GA2OPt")
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        format!("GA2Opt")
    }
}
#[cfg(test)]
mod tests {
    use crate::algorithms::TspSolver;
    use crate::metaheuristic::ga_two_opt::GA2Opt;
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
        let mut solver = GA2Opt::with_options(tsp, 100, 5, 0.7, 0.01, 500);

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