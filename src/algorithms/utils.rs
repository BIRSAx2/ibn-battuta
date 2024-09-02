use crate::algorithms::exact::branch_and_bound::BranchAndBound;
use crate::algorithms::exact::brute_force::BruteForce;
use crate::algorithms::heuristic::local_search::lin_kernighan::LinKernighan;
use crate::algorithms::heuristic::local_search::three_opt::ThreeOpt;
use crate::algorithms::heuristic::local_search::two_opt::TwoOpt;
use crate::algorithms::heuristic::nearest_neighbor::NearestNeighbor;
use crate::algorithms::metaheuristic::genetic_algorithm::GeneticAlgorithm;
use crate::algorithms::metaheuristic::simulated_annealing::SimulatedAnnealing;
use crate::algorithms::{heuristic, TspSolver};
use tspf::Tsp;

#[derive(Clone, Debug, PartialEq)]
pub enum Solver {
    BruteForce,
    BranchAndBound,
    NearestNeighbor,
    TwoOpt,
    ThreeOpt,
    Greedy,
    LinKernighan,
    SimulatedAnnealing,
    GeneticAlgorithm,
    AntColonyOptimization,
    AntColonySystem,
    RedBlackAntColonySystem,
}

impl Solver {
    pub fn create<'a>(&self, tsp: &'a Tsp, config: &SolverConfig) -> Box<dyn TspSolver + 'a> {
        match self {
            Solver::BruteForce => Box::new(BruteForce::new(tsp)),
            Solver::BranchAndBound => Box::new(BranchAndBound::new(tsp)),
            Solver::NearestNeighbor => Box::new(NearestNeighbor::new(tsp)),
            Solver::TwoOpt => Box::new(TwoOpt::new(tsp)),
            Solver::ThreeOpt => Box::new(ThreeOpt::new(tsp)),
            Solver::Greedy => Box::new(heuristic::greedy::Greedy::new(tsp)),
            Solver::LinKernighan => Box::new(LinKernighan::new(tsp)),
            Solver::SimulatedAnnealing => Box::new(SimulatedAnnealing::new(tsp)),
            Solver::GeneticAlgorithm => Box::new(GeneticAlgorithm::new(tsp)),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SolverConfig {
    ExactAlgorithm(ExactAlgorithmConfig),
    HeuristicAlgorithm(HeuristicAlgorithmConfig),
    MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig),
}

#[derive(Clone, Debug)]
pub enum ExactAlgorithmConfig {
    BellmanHeldKarp { max_iterations: u32 },
    BranchAndBound { max_iterations: u32 },
    BruteForce { max_iterations: u32 },
}

#[derive(Clone, Debug)]
pub enum HeuristicAlgorithmConfig {
    LocalSearch { base_solver: Solver, verbose: bool, max_iterations: usize },
}

#[derive(Clone, Debug)]
pub enum MetaheuristicAlgorithmConfig {
    SimulatedAnnealing {
        initial_temperature: f64,
        cooling_rate: f64,
        min_temperature: f64,
        max_iterations: usize,
    },
    GeneticAlgorithm {
        population_size: usize,
        tournament_size: usize,
        mutation_rate: f64,
        max_generations: usize,
    },
    AntColonyOptimization {
        alpha: f64,
        beta: f64,
        rho: f64,
        tau0: f64,
        q0: f64,
        num_ants: usize,
        evaporation_rate: f64,
        max_iterations: usize,
    },
    AntColonySystem {
        alpha: f64,
        beta: f64,
        rho: f64,
        tau0: f64,
        q0: f64,
        num_ants: usize,
        evaporation_rate: f64,
        elitism: f64,
        max_iterations: usize,
    },
    RedBlackAntColonySystem {
        alpha: f64,
        beta: f64,
        rho: f64,
        tau0: f64,
        q0: f64,
        num_ants: usize,
        num_red_ants: usize,
        num_black_ants: usize,
        evaporation_rate: f64,
        max_iterations: usize,
        elitism: f64,
    },
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig::HeuristicAlgorithm(HeuristicAlgorithmConfig::LocalSearch {
            base_solver: Solver::NearestNeighbor,
            verbose: false,
            max_iterations: 1000,
        })
    }
}

impl SolverConfig {
    pub fn new_bellman_held_karp(max_iterations: u32) -> Self {
        SolverConfig::ExactAlgorithm(ExactAlgorithmConfig::BellmanHeldKarp { max_iterations })
    }

    pub fn new_branch_and_bound(max_iterations: u32) -> Self {
        SolverConfig::ExactAlgorithm(ExactAlgorithmConfig::BranchAndBound { max_iterations })
    }

    pub fn new_brute_force(max_iterations: u32) -> Self {
        SolverConfig::ExactAlgorithm(ExactAlgorithmConfig::BruteForce { max_iterations })
    }

    pub fn new_local_search(base_solver: Solver, verbose: bool, max_iterations: usize) -> Self {
        SolverConfig::HeuristicAlgorithm(HeuristicAlgorithmConfig::LocalSearch { base_solver, verbose, max_iterations })
    }

    pub fn new_simulated_annealing(initial_temperature: f64, cooling_rate: f64, min_temperature: f64, max_iterations: usize) -> Self {
        SolverConfig::MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig::SimulatedAnnealing {
            initial_temperature,
            cooling_rate,
            min_temperature,
            max_iterations,
        })
    }

    pub fn new_genetic_algorithm(population_size: usize, tournament_size: usize, mutation_rate: f64, max_generations: usize) -> Self {
        SolverConfig::MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig::GeneticAlgorithm {
            population_size,
            tournament_size,
            mutation_rate,
            max_generations,
        })
    }

    pub fn new_ant_colony_optimization(alpha: f64, beta: f64, rho: f64, tau0: f64, q0: f64, num_ants: usize, evaporation_rate: f64, max_iterations: usize) -> Self {
        SolverConfig::MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig::AntColonyOptimization {
            alpha,
            beta,
            rho,
            tau0,
            q0,
            num_ants,
            evaporation_rate,
            max_iterations,
        })
    }

    pub fn new_ant_colony_system(alpha: f64, beta: f64, rho: f64, tau0: f64, q0: f64, num_ants: usize, evaporation_rate: f64, elitism: f64, max_iterations: usize) -> Self {
        SolverConfig::MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig::AntColonySystem {
            alpha,
            beta,
            rho,
            tau0,
            q0,
            num_ants,
            evaporation_rate,
            elitism,
            max_iterations,
        })
    }

    pub fn new_red_black_ant_colony_system(alpha: f64, beta: f64, rho: f64, tau0: f64, q0: f64, num_ants: usize, num_red_ants: usize, num_black_ants: usize, evaporation_rate: f64, max_iterations: usize, elitism: f64) -> Self {
        SolverConfig::MetaheuristicAlgorithm(MetaheuristicAlgorithmConfig::RedBlackAntColonySystem {
            alpha,
            beta,
            rho,
            tau0,
            q0,
            num_ants,
            num_red_ants,
            num_black_ants,
            evaporation_rate,
            max_iterations,
            elitism,
        })
    }
}