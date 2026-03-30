use crate::experimental::RedBlackACS;
use crate::{Solution, Tsp, TspSolver, TwoOpt};

/// This module implements the RBACS2Opt algorithm, which combines the Red-Black Ant Colony System (RBACS)
/// with the 2-opt local search algorithm for solving the Traveling Salesman Problem (TSP).
pub struct RBACS2Opt {
    tsp: Tsp,
    rbacs: RedBlackACS,
    last_solution: Solution,
}

impl RBACS2Opt {
    /// Creates a new instance of the RBACS2Opt algorithm with the specified options.
    ///
    /// # Arguments
    ///
    /// * `tsp` - The TSP instance to solve.
    /// * `alpha` - The pheromone importance factor.
    /// * `beta` - The heuristic importance factor.
    /// * `rho_red` - The pheromone evaporation rate for red edges.
    /// * `rho_black` - The pheromone evaporation rate for black edges.
    /// * `q0` - The probability of exploitation versus exploration.
    /// * `num_ants` - The number of ants in the colony.
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `candidate_list_size` - The size of the candidate list for local search.
    ///
    /// # Returns
    ///
    /// A new instance of the RBACS2Opt algorithm.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        tsp: Tsp,
        alpha: f64,
        beta: f64,
        rho_red: f64,
        rho_black: f64,
        q0: f64,
        num_ants: usize,
        max_iterations: usize,
        candidate_list_size: usize,
    ) -> RBACS2Opt {
        Self::with_options_and_seed(
            tsp,
            alpha,
            beta,
            rho_red,
            rho_black,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
            rand::random(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_options_and_seed(
        tsp: Tsp,
        alpha: f64,
        beta: f64,
        rho_red: f64,
        rho_black: f64,
        q0: f64,
        num_ants: usize,
        max_iterations: usize,
        candidate_list_size: usize,
        seed: u64,
    ) -> RBACS2Opt {
        let acs = RedBlackACS::new_with_seed(
            tsp.clone(),
            alpha,
            beta,
            rho_red,
            rho_black,
            q0,
            num_ants,
            max_iterations,
            candidate_list_size,
            seed,
        );

        RBACS2Opt {
            tsp,
            rbacs: acs,
            last_solution: Solution::default(),
        }
    }

    pub fn seed(&self) -> u64 {
        self.rbacs.seed()
    }

    fn optimize_tour(&self, base_tour: Vec<usize>) -> Solution {
        let local_optimum = TwoOpt::from(self.tsp.clone(), base_tour, false).solve();
        let perturbed = Self::double_bridge(&local_optimum.tour);
        if perturbed == local_optimum.tour {
            return local_optimum;
        }

        let iterated = TwoOpt::from(self.tsp.clone(), perturbed, false).solve();
        if iterated.length < local_optimum.length {
            iterated
        } else {
            local_optimum
        }
    }

    fn double_bridge(tour: &[usize]) -> Vec<usize> {
        let n = tour.len();
        if n < 8 {
            return tour.to_vec();
        }

        let q1 = n / 4;
        let q2 = n / 2;
        let q3 = (3 * n) / 4;

        if q1 == 0 || q1 == q2 || q2 == q3 || q3 >= n {
            return tour.to_vec();
        }

        let mut perturbed = Vec::with_capacity(n);
        perturbed.extend_from_slice(&tour[..q1]);
        perturbed.extend_from_slice(&tour[q3..]);
        perturbed.extend_from_slice(&tour[q2..q3]);
        perturbed.extend_from_slice(&tour[q1..q2]);
        perturbed
    }

    fn order_crossover(parent_a: &[usize], parent_b: &[usize]) -> Vec<usize> {
        let n = parent_a.len();
        if n < 4 {
            return parent_a.to_vec();
        }

        let start = n / 3;
        let end = (2 * n) / 3;
        let mut child = vec![usize::MAX; n];
        let mut used = vec![false; n];

        for idx in start..end {
            child[idx] = parent_a[idx];
            used[parent_a[idx]] = true;
        }

        let mut insert_idx = end % n;
        for &city in parent_b
            .iter()
            .cycle()
            .skip(end)
            .take(n)
        {
            if used[city] {
                continue;
            }
            while child[insert_idx] != usize::MAX {
                insert_idx = (insert_idx + 1) % n;
            }
            child[insert_idx] = city;
            insert_idx = (insert_idx + 1) % n;
        }

        child
    }
}

impl TspSolver for RBACS2Opt {
    /// Solves the TSP using the RBACS2Opt algorithm.
    ///
    /// # Returns
    ///
    /// A `Solution` struct containing the tour and its total cost.
    fn solve(&mut self) -> Solution {
        let _ = self.rbacs.solve();
        let [red_group, black_group] = self.rbacs.best_group_tours();
        let mut candidates = vec![
            self.optimize_tour(red_group.0.clone()),
            self.optimize_tour(black_group.0.clone()),
        ];

        let child_ab = Self::order_crossover(&red_group.0, &black_group.0);
        let child_ba = Self::order_crossover(&black_group.0, &red_group.0);
        candidates.push(self.optimize_tour(child_ab));
        candidates.push(self.optimize_tour(child_ba));

        self.last_solution = candidates
            .into_iter()
            .min_by(|lhs, rhs| lhs.length.total_cmp(&rhs.length))
            .unwrap();
        self.last_solution.clone()
    }

    /// Returns the tour of the TSP solution.
    ///
    /// # Returns
    ///
    /// A vector of node indices representing the tour.
    fn tour(&self) -> Vec<usize> {
        self.last_solution.tour.clone()
    }

    /// Calculates the cost between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from` - The starting node index.
    /// * `to` - The ending node index.
    ///
    /// # Returns
    ///
    /// The cost between the two nodes.
    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    /// Returns the name of the algorithm.
    ///
    /// # Returns
    ///
    /// A string representing the name of the algorithm.
    fn format_name(&self) -> String {
        "RBACS2Opt".to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::TspSolver;
    use crate::experimental::RBACS2Opt;
    use crate::TspBuilder;

    #[test]
    fn solves_simple_tsp_with_rbacs2opt() {
        let data = "
		NAME : simple
		TYPE : TSP
		DIMENSION : 4
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		  2 0.0 1.0
		  3 1.0 1.0
		  4 1.0 0.0
		EOF
		";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = RBACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.2, 0.9, 10, 1000, 15);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 4);
        assert!((solution.length - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_single_node_with_rbacs2opt() {
        let data = "
		NAME : single
		TYPE : TSP
		DIMENSION : 1
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		EOF
		";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = RBACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.2, 0.9, 10, 1000, 15);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 1);
        assert!((solution.length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn double_bridge_preserves_tour_membership() {
        let tour = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let perturbed = RBACS2Opt::double_bridge(&tour);

        assert_eq!(perturbed.len(), tour.len());
        let mut sorted_original = tour.clone();
        let mut sorted_perturbed = perturbed.clone();
        sorted_original.sort_unstable();
        sorted_perturbed.sort_unstable();
        assert_eq!(sorted_original, sorted_perturbed);
        assert_ne!(perturbed, tour);
    }

    #[test]
    fn order_crossover_preserves_tour_membership() {
        let parent_a = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let parent_b = vec![4, 5, 6, 7, 0, 1, 2, 3];
        let child = RBACS2Opt::order_crossover(&parent_a, &parent_b);

        assert_eq!(child.len(), parent_a.len());
        let mut sorted_child = child;
        sorted_child.sort_unstable();
        assert_eq!(sorted_child, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn handles_two_nodes_with_rbacs2opt() {
        let data = "
		NAME : two_nodes
		TYPE : TSP
		DIMENSION : 2
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		  2 1.0 0.0
		EOF
		";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = RBACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.2, 0.9, 10, 1000, 15);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 2);
        assert!((solution.length - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_non_euclidean_distances_with_rbacs2opt() {
        let data = "
		NAME : non_euclidean
		TYPE : TSP
		DIMENSION : 3
		EDGE_WEIGHT_TYPE: EXPLICIT
		EDGE_WEIGHT_FORMAT: FULL_MATRIX
		EDGE_WEIGHT_SECTION
		  0 2 9
		  1 0 6
		  15 7 0
		EOF
		";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut solver = RBACS2Opt::with_options(tsp, 0.1, 2.0, 0.1, 0.2, 0.9, 10, 1000, 15);
        let solution = solver.solve();

        assert_eq!(solution.tour.len(), 3);
        assert!((solution.length - 17.0).abs() < f64::EPSILON);
    }

    #[test]
    fn uses_seeded_runs_deterministically() {
        let data = "
		NAME : simple
		TYPE : TSP
		DIMENSION : 4
		EDGE_WEIGHT_TYPE: EUC_2D
		NODE_COORD_SECTION
		  1 0.0 0.0
		  2 0.0 1.0
		  3 1.0 1.0
		  4 1.0 0.0
		EOF
		";

        let tsp = TspBuilder::parse_str(data).unwrap();
        let mut lhs =
            RBACS2Opt::with_options_and_seed(tsp.clone(), 0.1, 2.0, 0.1, 0.2, 0.9, 10, 200, 15, 23);
        let mut rhs =
            RBACS2Opt::with_options_and_seed(tsp, 0.1, 2.0, 0.1, 0.2, 0.9, 10, 200, 15, 23);

        assert_eq!(lhs.seed(), 23);
        assert_eq!(lhs.solve(), rhs.solve());
    }
}
