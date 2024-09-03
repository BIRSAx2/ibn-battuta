use crate::algorithms::{Solution, TspSolver};
use std::f64;
use tspf::Tsp;


pub struct NearestNeighbor<'a> {
    tsp: &'a Tsp,
    visited: Vec<bool>,
    tour: Vec<usize>,
    cost: f64,
}

impl<'a> NearestNeighbor<'a> {
    pub fn new(tsp: &'a Tsp) -> Self {
        let n = tsp.dim();
        NearestNeighbor {
            tsp,
            visited: vec![false; n],
            tour: Vec::with_capacity(n),
            cost: 0.0,
        }
    }
}


impl TspSolver for NearestNeighbor<'_> {
    fn solve(&mut self) -> Solution {
        let n = self.tsp.dim();
        let mut current_city = 0;
        self.visited[current_city] = true;
        self.tour.push(current_city);

        for _ in 1..n {
            let mut next_city = None;
            let mut min_distance = f64::MAX;

            for city in 0..n {
                if !self.visited[city] {
                    let distance = self.tsp.weight(current_city, city) as f64;
                    if distance < min_distance {
                        min_distance = distance;
                        next_city = Some(city);
                    }
                }
            }

            if let Some(city) = next_city {
                self.visited[city] = true;
                self.tour.push(city);
                self.cost += min_distance;
                current_city = city;
            }
        }

        // Add the distance back to the start to complete the tour
        self.cost += self.tsp.weight(current_city, self.tour[0]) as f64;

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
    use super::*;
    use crate::algorithms::TspSolver;
    use tspf::TspBuilder;

    #[test]
    fn test_nearest_neighbor() {
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

        let mut solver = NearestNeighbor::new(&tsp);
        let solution = solver.solve();

        println!("{:?}", solution);
    }


    #[test]

    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        let size = tsp.dim();

        let mut solver = NearestNeighbor::new(&tsp);
        let solution = solver.solve();

        println!("{:?}", solution);
        assert_eq!(solution.tour.len(), size);
    }
}