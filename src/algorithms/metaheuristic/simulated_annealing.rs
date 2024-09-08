use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;

pub struct SimulatedAnnealing {
    tsp: Tsp,
    fire: Vec<usize>,
    best_path: Vec<usize>,
    best_length: f64,
    t0: f64,
    tend: f64,
    rate: f64,
    iter_x: Vec<usize>,
    iter_y: Vec<f64>,
}

impl SimulatedAnnealing {
    pub fn new(tsp: Tsp) -> Self {
        let num_city = tsp.dim();
        let mut sa =
            SimulatedAnnealing {
                tsp: tsp.clone(),
                fire: vec![],
                best_path: vec![],
                best_length: f64::MAX,
                t0: 4000.0,
                tend: 1e-3,
                rate: 0.9995,
                iter_x: vec![0],
                iter_y: vec![0.0],
            };

        let fire = sa.greedy_init(&tsp, 100, num_city);
        let init_pathlen = sa.compute_pathlen(&fire, &tsp);
        sa.fire = fire.clone();
        sa.best_path = fire;
        sa.best_length = init_pathlen;
        sa.iter_y[0] = init_pathlen;
        sa
    }

    fn greedy_init(&self, tsp: &Tsp, num_total: usize, num_city: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut result = Vec::new();

        for _ in 0..num_total {
            let mut rest: Vec<usize> = (0..num_city).collect();
            let mut current = if result.len() < num_city {
                result.len()
            } else {
                rng.gen_range(0..num_city)
            };

            let mut result_one = vec![current];
            rest.retain(|&x| x != current);

            while !rest.is_empty() {
                let (tmp_choose, _) = rest.iter()
                    .map(|&x| (x, self.cost(current, x)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();

                current = tmp_choose;
                result_one.push(tmp_choose);
                rest.retain(|&x| x != tmp_choose);
            }

            result.push(result_one);
        }

        let path_lens: Vec<f64> = result.iter()
            .map(|path| self.compute_pathlen(path, tsp))
            .collect();

        result[path_lens.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0
            ].clone()
    }

    fn compute_pathlen(&self, path: &Vec<usize>, _tsp: &Tsp) -> f64 {
        let mut result = self.cost(*path.last().unwrap(), path[0]);
        for i in 0..path.len() - 1 {
            result += self.cost(path[i], path[i + 1]);
        }
        result
    }

    fn get_new_fire(&self) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut new_fire = self.fire.clone();
        let (a, b) = (rng.gen_range(0..new_fire.len()), rng.gen_range(0..new_fire.len()));
        new_fire[a.min(b)..=a.max(b)].reverse();
        new_fire
    }

    fn eval_fire(&self, raw: &Vec<usize>, get: &Vec<usize>, temp: f64) -> (Vec<usize>, f64) {
        let len1 = self.compute_pathlen(raw, &self.tsp);
        let len2 = self.compute_pathlen(get, &self.tsp);
        let dc = len2 - len1;
        let p = f64::max(1e-1, f64::exp(-dc / temp));

        if len2 < len1 || rand::random::<f64>() <= p {
            (get.clone(), len2)
        } else {
            (raw.clone(), len1)
        }
    }

    pub fn sa(&mut self) -> (f64, Vec<usize>) {
        let mut count = 0;
        let mut t = self.t0;

        while t > self.tend {
            count += 1;
            let tmp_new = self.get_new_fire();
            let (new_fire, file_len) = self.eval_fire(&self.best_path, &tmp_new, t);

            self.fire = new_fire;

            if file_len < self.best_length {
                self.best_length = file_len;
                self.best_path = self.fire.clone();
            }

            t *= self.rate;

            self.iter_x.push(count);
            self.iter_y.push(self.best_length);
        }

        (self.best_length, self.best_path.clone())
    }
}

impl TspSolver for SimulatedAnnealing {
    fn solve(&mut self) -> Solution {
        let (best_length, best_path) = self.sa();
        Solution {
            tour: best_path,
            total: best_length,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.fire.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        "SimulatedAnnealing".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::TspSolver;
    use crate::TspBuilder;

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
        let mut solver = SimulatedAnnealing::new(tsp.clone());
        let solution = solver.solve();

        println!("{:?}", solution);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/bier127.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let sol = test_instance(tsp);
        let best_known = 118282.0;
        let gap = (sol.total - best_known) / best_known;
        println!("Gap: {:.2}%", gap * 100.0);
    }

    #[test]
    fn test_gr120() {
        let path = "data/tsplib/st70.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();
        test_instance(tsp);
    }

    fn test_instance(tsp: Tsp) -> Solution {
        let size = tsp.dim();
        let mut solver = SimulatedAnnealing::new(tsp);
        let solution = solver.solve();
        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
        solution
    }
}