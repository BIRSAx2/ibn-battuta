use std::collections::HashMap;

use getset::Getters;

use crate::parser::Tsp;

#[derive(Debug, Getters)]
pub struct SparseMatrix {
    n: usize,
    m: usize,
    data: HashMap<(usize, usize), f64>,
}


impl SparseMatrix {
    pub fn new(n: usize, m: usize) -> SparseMatrix {
        let data = HashMap::new();

        SparseMatrix { n, m, data }
    }

    pub fn get(&self, i: usize, j: usize) -> Option<&f64> {
        if i > j {
            self.data.get(&(i, j))
        } else {
            self.data.get(&(j, i))
        }
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) -> Option<f64> {
        if i > j {
            self.data.insert((i, j), value)
        } else {
            self.data.insert((j, i), value)
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn neighbours_of(&self, index: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.n).collect();
        v.retain(|&x| x != index);
        v
    }

    // Create a SparseMatrix from a Tsp struct
    pub fn from_tsp(tsp: &Tsp) -> Self {
        let n = tsp.dim();
        let m = tsp.dim();
        let mut data = HashMap::new();

        // Populate the data based on the edge weights in the Tsp struct
        for i in 0..n {
            for j in 0..m {
                let weight = tsp.weight(i, j);

                // Only include non-zero weights in the sparse matrix
                if weight != 0.0 {
                    data.insert((i, j), weight);
                }
            }
        }

        SparseMatrix { n, m, data }
    }
}