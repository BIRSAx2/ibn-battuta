#[derive(Clone, Debug, PartialEq, Copy)]
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
	GeneticAlgorithm2Opt,
	AntSystem,
	AntColonySystem,
	AntColonySystem2Opt,
	RedBlackAntColonySystem,
	RedBlackAntColonySystem2Opt,
	SimulatedAnnealing2Opt,
}

