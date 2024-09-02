use crate::algorithms::utils::SolverConfig;
use crate::algorithms::{Solution, TspSolver};
use tspf::Tsp;

pub struct BranchAndBound<'a> {
    tsp: &'a Tsp,
    best_tour: Vec<usize>,
    best_cost: f64,
}

impl<'a> BranchAndBound<'a> {
    pub fn new(tsp: &'a Tsp) -> Self {
        BranchAndBound {
            tsp,
            best_tour: vec![],
            best_cost: f64::INFINITY,
        }
    }

    fn branch_and_bound(&mut self, current_tour: Vec<usize>, current_cost: f64, visited: Vec<bool>) {
        if current_tour.len() == self.tsp.dim() {
            if current_cost < self.best_cost {
                self.best_tour = current_tour.clone();
                self.best_cost = current_cost;
            }
            return;
        }

        let current_node = current_tour.last().unwrap();
        for next_node in 0..self.tsp.dim() {
            if visited[next_node] {
                continue;
            }

            let new_cost = current_cost + self.tsp.weight(*current_node, next_node);
            if new_cost >= self.best_cost {
                continue;
            }

            let mut new_visited = visited.clone();
            new_visited[next_node] = true;
            let mut new_tour = current_tour.clone();
            new_tour.push(next_node);

            self.branch_and_bound(new_tour, new_cost, new_visited);
        }
    }

    pub fn run(&mut self) {
        let mut visited = vec![false; self.tsp.dim()];
        visited[0] = true;
        self.branch_and_bound(vec![0], 0.0, visited);
    }
}

impl TspSolver for BranchAndBound<'_> {
    fn solve(&mut self, _options: &SolverConfig) -> Solution {
        self.run();
        Solution::new(self.best_tour.iter().map(|&i| i as usize).collect(), self.best_cost)
    }
    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }
}


#[cfg(test)]
mod tests {
    use crate::algorithms::exact::branch_and_bound::BranchAndBound;
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

        let mut solver = BranchAndBound::new(&tsp);
        let mut options = SolverConfig::new_branch_and_bound(1000);

        let solution = solver.solve(&mut options);

        assert_eq!(solution.tour.len(), 5);
        assert!((solution.total - 13.646824151749852).abs() < f64::EPSILON);
    }
}