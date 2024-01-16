// Implementation of the Ant Colony System algorithm as described in the paper:
// Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem by Marco Dorigo et al



use std::collections::HashSet;

use getset::{Getters, MutGetters, Setters};
use rand::Rng;

use crate::parser;
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
}

impl AntColonySystem {
    fn new(tsp: Tsp, num_ants: usize, alpha: f64, beta: f64, rho: f64, tau0: f64, q0: f64) -> Self {
        let pheromone_matrix = vec![vec![tau0; tsp.dim() + 1]; tsp.dim() + 1];
        let distance_matrix = build_distance_matrix(&tsp);

        let mut ants = Vec::with_capacity(num_ants);
        for _ in 0..num_ants {
            ants.push(Ant {
                first_node: 0,
                current_node: 0,
                unvisited_nodes: (1..tsp.dim()).collect(),
                current_tour: Vec::with_capacity(tsp.dim()),
                tour_length: 0.0,
            });
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
        }
    }


    fn initialize_ants(&mut self) {
        // pheromone matrix is already initialized

        // generate a random number uniformly distributed in [0, distance_matrix.len()]
        let mut rng = rand::thread_rng();
        for ant in self.ants.iter_mut() {
            let start_node = rng.gen_range(0..self.distance_matrix.len());
            ant.current_node = start_node;
            ant.first_node = start_node;
            ant.unvisited_nodes = (1..self.distance_matrix.len()).collect();
            ant.unvisited_nodes.remove(&start_node);
            ant.current_tour.clear();
            ant.current_tour.push(start_node);
            ant.tour_length = 0.0;
        }
    }
    pub fn solve(&mut self, max_iterations: usize) {
        // Initialization phase
        self.initialize_ants();

        for j in 0..max_iterations {
            // this is the phase in which ants build their tours
            for i in 1..self.distance_matrix.len() {
                if i < self.distance_matrix.len() - 2 {
                    let mut ants: Vec<Ant> = self.ants.to_vec();
                    for mut ant in ants.iter_mut() {
                        let next_node = self.pseudo_random_proportional_rule(&ant);
                        ant.unvisited_nodes.remove(&next_node);
                        ant.current_tour.push(next_node);
                        ant.tour_length += self.distance_matrix[ant.current_node][next_node];
                    }
                    self.ants = ants;
                } else {
                    // last node
                    for ant in self.ants.iter_mut() {
                        ant.tour_length += self.distance_matrix[ant.current_node][ant.first_node];
                        ant.current_node = ant.first_node;
                    }
                }
                // local pheromone update
                for ant in self.ants.iter_mut() {
                    let current_node = ant.current_node;
                    let next_node = *ant.current_tour.last().unwrap();
                    self.pheromone_matrix[current_node][next_node] = (1.0 - self.alpha) * self.pheromone_matrix[current_node][next_node] + self.alpha * self.tau0;
                    ant.current_node = next_node;
                }
            }

            let best_ant = &self.ants.iter().min_by(|a, b| a.tour_length.total_cmp(&b.tour_length)).unwrap();
            for i in 1..best_ant.current_tour.len() - 1 {
                let nodea = best_ant.current_tour[i];
                let nodeb = best_ant.current_tour[i + 1];
                // this is the issue
                self.pheromone_matrix[nodea][nodeb] = (1.0 - self.alpha) * self.pheromone_matrix[nodea][nodeb] + self.alpha * best_ant.tour_length.powi(-1);
            }

            println!("Tour size: {}", best_ant.current_tour.len());
            // use as_str to get a `&str` from a String to avoid copying the string
            let uniques: HashSet<usize> = best_ant.current_tour.iter()
                .map(|c| *c)
                .collect();
            println!("Duplicates: {}", best_ant.current_tour.len() - uniques.len());

            println!("Tour: {:?}", best_ant.current_tour);
            println!("End of iteration {}, best tour:  {}", j, best_ant.tour_length);

            self.initialize_ants();
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
                return 1.0;
            }
            distance(current_node, next_node) * eta(current_node, next_node).powf(self.beta)
        };


        let random_proportional_rule = |current_node, next_node| -> f64 {
            if !ant.unvisited_nodes.contains(&next_node) {
                return 0.0;
            }

            let numerator = transition_rule(current_node, next_node);
            let denominator = ant.unvisited_nodes.iter().map(|node| transition_rule(current_node, *node)).sum::<f64>();

            numerator / denominator
        };

        if q <= self.q0 {
            // exploitation
            ant.unvisited_nodes.iter().max_by(|a, b| {
                transition_rule(ant.current_node, **a).partial_cmp(&transition_rule(ant.current_node, **b)).unwrap()
            }).unwrap().clone()
        } else {
            // exploration
            ant.unvisited_nodes.iter().max_by(|a, b| {
                random_proportional_rule(ant.current_node, **a).partial_cmp(&random_proportional_rule(ant.current_node, **b)).unwrap()
            }).unwrap().clone()
        }
    }
}

fn build_distance_matrix(tsp: &Tsp) -> Vec<Vec<f64>> {
    let mut distance_matrix = vec![vec![0.0; tsp.dim() + 1]; tsp.dim() + 1];
    let node_coordinates = tsp.node_coords();


    for (id, node) in node_coordinates.iter() {
        for (id2, node2) in node_coordinates.iter() {
            // TODO: change this to use the appropriate metric
            distance_matrix[*id][*id2] = parser::metric::euc_2d(node.coordinates(), node2.coordinates());
        }
    }

    distance_matrix
}


#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::algorithms::{nearest_neighbor, SolverOptions};
    use crate::algorithms::ant_colony::acs::AntColonySystem;
    use crate::parser::TspBuilder;
    use crate::Point;

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
        let mut acs = AntColonySystem::new(tsp, 10, 0.1, 2.0, 0.1, 1.0 / (node_coords.len() as f64 * length_nn), 0.9);
        acs.solve(100);
    }
}