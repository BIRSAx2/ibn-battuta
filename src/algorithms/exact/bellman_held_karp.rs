use crate::algorithms::{Solution, TspSolver};
use std::f64;
use crate::Tsp;

pub struct BellmanHeldKarp {
    tsp: Tsp,
    opt: Vec<Vec<Option<f64>>>,
    best_tour: Vec<usize>,
    best_cost: f64,
}
impl BellmanHeldKarp {
    pub fn new(tsp: Tsp) -> Self {
        let n = tsp.dim();
        let opt = vec![vec![None; 1 << (n - 1)]; n - 1];
        BellmanHeldKarp {
            tsp,
            opt,
            best_tour: Vec::with_capacity(n),
            best_cost: f64::INFINITY,
        }
    }
}

impl BellmanHeldKarp {
    pub fn bellman_held_karp(&mut self) {
        let n = self.tsp.dim();
        let dist = |i: usize, j: usize| self.tsp.weight(i, j);

        // Initialize the base cases
        for i in 0..n - 1 {
            self.opt[i][1 << i] = Some(dist(i, n - 1));
        }

        // Iterate over subsets of increasing size
        for size in 2..n {
            for s in 1..(1 << (n - 1))
            {
                if (s as i32).count_ones() as usize == size {
                    for t in 0..n - 1 {
                        if (s & (1 << t)) != 0 {
                            let mut min_cost = f64::INFINITY;
                            let prev_s = s & !(1 << t);
                            for q in 0..n - 1 {
                                if (prev_s & (1 << q)) != 0 {
                                    if let Some(cost) = self.opt[q][prev_s] {
                                        min_cost = f64::min(min_cost, cost + dist(q, t));
                                    }
                                }
                            }
                            self.opt[t][s] = Some(min_cost);
                        }
                    }
                }
            }
        }

        // Calculate the minimum cost to complete the tour
        for t in 0..n - 1 {
            if let Some(cost) = self.opt[t][(1 << (n - 1)) - 1] {
                let final_cost = cost + dist(t, n - 1);
                if final_cost < self.best_cost {
                    self.best_cost = final_cost;
                    self.best_tour = self.build_tour(t, (1 << (n - 1)) - 1);
                }
            }
        }
    }

    fn build_tour(&self, mut last: usize, mut s: usize) -> Vec<usize> {
        let mut tour = vec![last];
        for _ in 1..self.tsp.dim() - 1 {
            for i in 0..self.tsp.dim() - 1 {
                if s & (1 << i) != 0 {
                    if let Some(cost) = self.opt[i][s & !(1 << last)] {
                        if cost + self.tsp.weight(i, last) == self.opt[last][s].unwrap() {
                            tour.push(i);
                            s &= !(1 << last);
                            last = i;
                            break;
                        }
                    }
                }
            }
        }
        tour.push(self.tsp.dim() - 1);
        tour.reverse();
        tour
    }
}

impl TspSolver for BellmanHeldKarp {
    fn solve(&mut self) -> Solution {
        self.bellman_held_karp();
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
    use crate::algorithms::exact::bellman_held_karp::BellmanHeldKarp;
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

        let mut solver = BellmanHeldKarp::new(tsp);

        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 5);
        assert!((solution.total - 13.646824151749852).abs() < f64::EPSILON);
    }
}