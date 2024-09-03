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
