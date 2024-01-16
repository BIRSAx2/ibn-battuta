use std::collections::HashSet;

use getset::{Getters, MutGetters, Setters};
use rand::Rng;

use crate::{parser, Point};
use crate::parser::Tsp;

#[derive(Debug, Clone, MutGetters, Setters, Getters)]
struct Ant {
    first_node: usize,
    current_node: usize,
    unvisited_nodes: HashSet<usize>,
    current_tour: Vec<usize>,
    tour_length: f64,
}


#[derive(Debug)]
struct AntColonySystem {
    ants: Vec<Ant>,
    pheromone_matrix: Vec<Vec<f64>>,
    distance_matrix: Vec<Vec<f64>>,
    alpha: f64,
    rho: f64,
    beta: f64,
    tau0: f64,
    q0: f64,
    tsp: Tsp,
}

fn build_distance_matrix(tsp: &Tsp) -> Vec<Vec<f64>> {
    let mut distance_matrix = vec![vec![0.0; tsp.dim()]; tsp.dim()];
    let node_coordinates = tsp.node_coords();

    for (id, node) in node_coordinates.iter() {
        for (id2, node2) in node_coordinates.iter() {
            // node IDs starts from 1, while in the algorithm it's zero-based
            distance_matrix[*id - 1][*id2 - 1] = parser::metric::euc_2d(node.coordinates(), node2.coordinates());
        }
    }
    distance_matrix
}

impl AntColonySystem {
    fn new(tsp: Tsp, num_ants: usize, alpha: f64, beta: f64, rho: f64, tau0: f64, q0: f64) -> Self {
        // initialize pheromone matrix
        let pheromone_matrix = vec![vec![tau0; tsp.dim()]; tsp.dim()];
        let distance_matrix = build_distance_matrix(&tsp);

        let mut ants = Vec::with_capacity(num_ants);

        for _ in 0..num_ants {
            ants.push(Ant {
                first_node: 0,
                current_node: 0,
                unvisited_nodes: (0..tsp.dim()).collect(),
                current_tour: Vec::with_capacity(tsp.dim()),
                tour_length: 0.0,
            })
        }

        AntColonySystem {
            ants,
            pheromone_matrix,
            distance_matrix,
            alpha,
            rho,
            beta,
            tau0,
            q0,
            tsp,
        }
    }


    fn initialize_ants(&mut self) {
        let mut rng = rand::thread_rng();
        for ant in self.ants.iter_mut() {
            let start_node = rng.gen_range(0..self.distance_matrix.len());
            ant.current_node = start_node;
            ant.first_node = start_node;
            ant.unvisited_nodes = (0..self.distance_matrix.len()).collect();
            ant.unvisited_nodes.remove(&start_node);
            ant.current_tour.clear();
            ant.current_tour.push(start_node);
            ant.tour_length = 0.0
        }
    }


    fn pseudo_random_proportional_rule(&self, ant: &Ant) -> usize {
        let q = rand::thread_rng().gen_range(0.0..1.0);

        let distance = |current_node, next_node| -> f64 {
            let dm: &Vec<f64> = self.distance_matrix.get(current_node).unwrap();
            *dm.get(next_node).unwrap()
        };


        let eta = |current_node, next_node| -> f64 {
            1.0 / distance(current_node, next_node)
        };

        let transition_rule = |current_node, next_node| -> f64 {
            if distance(current_node, next_node) == 0.0 {
                // edge case where two cities have the same coordinates. Ex a280.tsp
                return 1.0;
            }
            distance(current_node, next_node) * eta(current_node, next_node).powf(self.beta)
        };

        let random_proportional_rule = |current_node, next_node| -> f64 {
            if !ant.unvisited_nodes.contains(&next_node) {
                return 0.0;
            }

            let numerator = transition_rule(current_node, next_node);
            let denominator = ant.unvisited_nodes.iter().map(|other_node| transition_rule(current_node, *other_node)).sum::<f64>();

            numerator / denominator
        };

        if q <= self.q0 {

            // exploration
            ant.unvisited_nodes.iter().max_by(|a, b| {
                transition_rule(ant.current_node, **a).partial_cmp(&transition_rule(ant.current_node, **b)).unwrap()
            }).unwrap().clone()
        } else {
            // biased exploration
            ant.unvisited_nodes.iter().max_by(|a, b| {
                random_proportional_rule(ant.current_node, **a).partial_cmp(&random_proportional_rule(ant.current_node, **b)).unwrap()
            }).unwrap().clone()
        }
    }


    pub fn solve(&mut self, max_iterations: usize) -> (f64, Vec<usize>) {
        let mut lowest_found = f64::MAX;
        let mut lowest_path: Vec<usize> = Vec::with_capacity(self.tsp.dim());
        for _ in 0..max_iterations
        {
            self.initialize_ants();
            // tour phase building

            for i in 0..self.distance_matrix.len() {
                let mut next_nodes: Vec<usize> = vec![0; self.ants.len()];
                if i < self.distance_matrix.len() - 1 {
                    for k in 0..self.ants.len() {
                        next_nodes[k] = self.pseudo_random_proportional_rule(&self.ants[k]);
                        self.ants[k].unvisited_nodes.remove(&next_nodes[k]);
                        self.ants[k].current_tour.push(next_nodes[k]);
                    }
                } else {
                    for k in 0..self.ants.len() {
                        next_nodes[k] = self.ants[k].first_node;
                        self.ants[k].current_tour.push(next_nodes[k]);
                    }
                }
                for k in 0..self.ants.len() {
                    let rk = self.ants[k].current_node;
                    let sk = next_nodes[k];
                    self.pheromone_matrix[rk][sk] = (1.0 - self.rho) * self.pheromone_matrix[rk][sk] + self.rho * self.tau0;
                    self.ants[k].current_node = sk;
                }
            }

            let mut current_best = f64::MAX;
            let mut current_best_tour = self.ants.get(0).unwrap().current_tour.clone();
            let mut node_coords = self.tsp.node_coords().iter().map(|(_, node)| node.clone()).collect::<Vec<Point>>();
            for k in 0..self.ants.len() {
                let mut dist = 0.0;

                for i in 0..self.ants[k].current_tour.len() - 1 {
                    let a = current_best_tour[i];
                    let b = current_best_tour[i + 1];
                    dist += self.distance_matrix[a][b];
                }
                if dist < current_best {
                    current_best_tour = self.ants[k].current_tour.clone();
                    current_best = dist;
                }
            }

            println!("Current best tour: {}", current_best);

            if current_best < lowest_found {
                lowest_path = current_best_tour.clone();
                lowest_found = current_best;
            }

            // update edges belonging to the best tour found
            for i in 0..current_best_tour.len() - 1 {
                let rk = current_best_tour[i];
                let sk = current_best_tour[i + 1];
                self.pheromone_matrix[rk][sk] = (1.0 - self.alpha) * self.pheromone_matrix[rk][sk] + self.alpha * (1.0 / current_best);
            }
        }

        return (lowest_found, lowest_path);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::algorithms::{nearest_neighbor, SolverOptions};
    use crate::parser::TspBuilder;
    use crate::Point;

    use super::*;

    #[test]
    fn test() {
        let path = Path::new("data/tsplib/berlin52.tsp");
        let tsp = TspBuilder::parse_path(path).unwrap();

        let options = SolverOptions::default();
        let mut node_coords = tsp.node_coords().iter().map(|(_, node)| node.clone()).collect::<Vec<Point>>();

        node_coords.sort_by(|a, b| a.id().cmp(&b.id()));
        let solution = nearest_neighbor::solve(&node_coords, &options);
        println!("NN sol: {}", solution.total);
        let length_nn = solution.total;
        // let mut acs = AntColonySystem::new(tsp, 30, 9.0, 12.0, 0.15, 0.0001, 0.2);
        let mut acs = AntColonySystem::new(tsp, 10, 0.1, 2.0, 0.1, 1.0 / (node_coords.len() as f64 * length_nn), 0.9);

        let (best, tour) = acs.solve(100);

        // 8182.1915557256725

        println!("Best tour found: {:?}", best);
        println!("Best tour {:?}", tour)
    }
}

