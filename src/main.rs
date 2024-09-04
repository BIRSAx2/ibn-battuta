use ibn_battuta::algorithms::utils::Solver;
use ibn_battuta::algorithms::*;
use ibn_battuta::parser::TspBuilder;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Define a struct to hold TSP instance data
#[derive(Clone, Debug)]
pub struct TspInstance {
    pub path: String,
    pub best_known: f64,
}

// Define a struct to hold benchmark results
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkResult {
    pub instance_name: String,
    pub algorithm_name: String,
    pub execution_time: Duration,
    pub total_cost: f64,
    pub best_known: f64,
    pub solution_quality: f64,
    pub solution: Vec<usize>,
}

fn run_parallel_benchmarks(
    instances: &[TspInstance],
    algorithm: Solver,
    params: Vec<f64>,
) -> Vec<BenchmarkResult> {
    let instances = Arc::new(<[TspInstance]>::to_vec(&instances));
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for instance in instances.iter() {
        let instance = instance.clone();
        let results = Arc::clone(&results);
        let instances = Arc::clone(&instances);
        let params = params.clone();
        let handle = thread::spawn(move || {
            let result = run_benchmark_multiple(&instance, algorithm.clone(), params.clone(), 5);
            println!("Benchmarking for instance: {}", instance.path);
            results.lock().unwrap().push(result);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}

fn run_parallel_grid_search(
    instance: &TspInstance,
    algorithm: Solver,
    params: Vec<Vec<f64>>,
) -> Vec<BenchmarkResult> {
    todo!("Implement run_parallel_grid_search function");
    // let results = Arc::new(Mutex::new(Vec::new()));
    // let tournament_sizes = Arc::new(params.get(0).to_vec());
    // let mutation_rates = Arc::new(params.get(1).to_vec());
    // let max_generations = Arc::new(params.get(2).to_vec());
    // let mut handles = vec![];
    //
    // for &pop_size in 0..1 {
    //     let instance = instance.clone();
    //     let results = Arc::clone(&results);
    //     let tournament_sizes = Arc::clone(&tournament_sizes);
    //     let mutation_rates = Arc::clone(&mutation_rates);
    //     let max_generations = Arc::clone(&max_generations);
    //     let handle = thread::spawn(move || {
    //         println!("Running benchmark with pop_size: {}", pop_size);
    //         for &tourn_size in tournament_sizes.iter() {
    //             for &mut_rate in mutation_rates.iter() {
    //                 for &max_gen in max_generations.iter() {
    //                     println!("Running benchmark with pop_size: {}, tourn_size: {}, mut_rate: {}, max_gen: {}", pop_size, tourn_size, mut_rate, max_gen);
    //                     let result = run_benchmark_multiple(&instance, algorithm, params, 5);
    //                     results.lock().unwrap().push(result);
    //                 }
    //             }
    //         }
    //     });
    //     handles.push(handle);
    // }
    //
    // for handle in handles {
    //     handle.join().unwrap();
    // }
    // Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}


fn benchmark(solver: Solver, params: Vec<f64>) {
    let instances_names = vec![
        ("gr17", 2085f64),
        ("gr21", 2707f64),
        ("kroA100", 21285.45),
        ("bier127", 118282.0),
        ("gr48", 5046.0),
        ("brazil58", 25395.0),
        ("gr120", 6942.0),
        ("gr137", 69853.0),
        ("gr202", 40160.0),
        ("gr666", 294358.0),
    ];
    // Define TSP instances
    let instances: Vec<TspInstance> = instances_names
        .iter()
        .map(|(name, best_known)| TspInstance {
            path: format!("data/tsplib/{}.tsp", name),
            best_known: *best_known as f64,
        }).collect();

    // Benchmark on different instances in parallel
    println!("Benchmarking on different instances:");
    let instance_results = run_parallel_benchmarks(&instances, solver, Vec::from(params));
    for result in &instance_results {
        print_benchmark_result(result);
    }

    // Save instance benchmark results to CSV
    save_results_to_csv(&instance_results, format!("{}-Instance-Benchmark.csv", solver).as_str());

    // // Grid search on gr48 instance
    // println!("\nGrid search on gr48 instance:");
    // let gr48 = &instances[3];
    // let population_sizes = vec![100, 200];
    // let tournament_sizes = vec![5, 10];
    // let mutation_rates = vec![0.01, 0.02];
    // let max_generations = vec![500, 1000];
    //
    // let grid_search_results = run_parallel_grid_search(gr48, &population_sizes, &tournament_sizes, &mutation_rates, &max_generations);
    //
    // let best_result = grid_search_results.iter().min_by(|a, b| a.solution_quality.partial_cmp(&b.solution_quality).unwrap()).unwrap();
    //
    // println!("Best configuration for gr48:");
    // print_benchmark_result(best_result);
    //
    // // Save grid search results to CSV
    // save_results_to_csv(&grid_search_results, "ga_grid_search.csv");
}


fn build_solver<'a>(instance: String, algorithm: Solver, params: &Vec<f64>) -> Box<dyn TspSolver + 'a> {
    let tsp = TspBuilder::parse_path(&instance).unwrap();
    match algorithm {
        Solver::GeneticAlgorithm => {
            let population_size = params[0] as usize;
            let tournament_size = params[1] as usize;
            let mutation_rate = params[2];
            let max_generations = params[3] as usize;
            Box::new(GeneticAlgorithm::with_options(tsp, population_size, tournament_size, mutation_rate, max_generations))
        }
        Solver::NearestNeighbor => {
            Box::new(NearestNeighbor::new(tsp))
        }
        Solver::TwoOpt => {
            let mut nn = NearestNeighbor::new(tsp.clone());
            let base_tour = nn.solve().tour;


            Box::new(TwoOpt::from(tsp, base_tour, false))
        }
        Solver::LinKernighan => {
            let mut nn = NearestNeighbor::new(tsp.clone());
            let base_tour = nn.solve().tour;

            Box::new(LinKernighan::with_options(tsp, base_tour, true, 1000))
        }
        Solver::SimulatedAnnealing => {
            let initial_temperature = params[0];
            let cooling_rate = params[1];
            let min_temperature = params[2];
            let max_iterations = params[3] as usize;
            Box::new(SimulatedAnnealing::with_options(tsp, initial_temperature, cooling_rate, min_temperature, max_iterations))
        }
        Solver::AntColonySystem => {
            let mut nn = NearestNeighbor::new(tsp.clone());
            let base_tour = nn.solve().total;
            let n = tsp.dim();
            let tau0 = 1.0 / (n as f64 * base_tour as f64);

            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let q0 = params[3];
            let num_ants = params[4] as usize;
            let max_iterations = params[5] as usize;
            Box::new(AntColonySystem::with_options(tsp, alpha, beta, rho, tau0, q0, num_ants, max_iterations))
        }
        Solver::RedBlackAntColonySystem => {
            let mut nn = NearestNeighbor::new(tsp.clone());
            let base_tour = nn.solve().total;
            let n = tsp.dim();
            let tau0 = 1.0 / (n as f64 * base_tour as f64);

            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let q0 = params[3];
            let num_ants = params[4] as usize;
            let max_iterations = params[5] as usize;
            Box::new(RedBlackACS::with_options(tsp, alpha, beta, rho, tau0, q0, num_ants, max_iterations))
        }
        _ => unimplemented!(),
    }
}
fn run_benchmark_multiple(
    instance: &TspInstance,
    algorithm: Solver,
    params: Vec<f64>,
    num_runs: usize,
) -> BenchmarkResult {
    let mut best_result = None;
    let mut best_quality = f64::INFINITY;
    let mut total_duration = Duration::new(0, 0);

    for _ in 0..num_runs {
        let start = Instant::now();
        let tsp = Arc::new(TspBuilder::parse_path(&instance.path).unwrap());
        let mut solver = {
            build_solver(instance.path.clone(), algorithm, &params)
        };
        let solution = solver.solve();
        let duration = start.elapsed();
        total_duration += duration;

        let quality = (solution.total - instance.best_known) / instance.best_known * 100.0;
        if quality < best_quality {
            best_quality = quality;
            best_result = Some(BenchmarkResult {
                instance_name: tsp.name().to_string(),
                algorithm_name: format!(
                    "{}",
                    solver
                ),
                execution_time: duration,
                total_cost: solution.total,
                best_known: instance.best_known,
                solution_quality: quality,
                solution: solution.tour,
            });
        }
    }

    let mut result = best_result.unwrap();
    result.execution_time = total_duration / num_runs as u32;  // Average execution time
    result
}

fn run_benchmark(
    instance: &TspInstance,
    population_size: usize,
    tournament_size: usize,
    mutation_rate: f64,
    max_generations: usize,
) -> BenchmarkResult {
    let start = Instant::now();
    let tsp = TspBuilder::parse_path(&instance.path).unwrap();
    let mut solver = GeneticAlgorithm::with_options(
        tsp,
        population_size,
        tournament_size,
        mutation_rate,
        max_generations,
    );
    let solution = solver.solve();
    let duration = start.elapsed();

    let tsp = Box::new(TspBuilder::parse_path(&instance.path).unwrap());

    BenchmarkResult {
        instance_name: tsp.name().to_string(),
        algorithm_name: format!(
            "GA ({};{};{:.3};{})",
            population_size, tournament_size, mutation_rate, max_generations
        ),
        execution_time: duration,
        total_cost: solution.total,
        best_known: instance.best_known,
        solution_quality: (solution.total - instance.best_known) / instance.best_known * 100.0,
        solution: solution.tour,
    }
}

fn print_benchmark_result(result: &BenchmarkResult) {
    println!(
        "Instance: {}, Algorithm: {}, Time: {:?}, Found Tour Length: {:.2}, Best known Length: {:.2}, Gap: {:.2}%",
        result.instance_name,
        result.algorithm_name,
        result.execution_time,
        result.total_cost,
        result.best_known,
        result.solution_quality
    );
}

fn save_results_to_csv(results: &[BenchmarkResult], filename: &str) {
    let mut file = File::create(filename).expect("Unable to create file");

    // Write header
    writeln!(file, "Instance,Algorithm,Time (ms),Found Tour Length,Best Known Length,Gap (%),Solution").expect("Unable to write to file");

    // Write data
    for result in results {
        writeln!(
            file,
            "{},{},{},{:.2},{:.2},{:.2},\"{}\"",
            result.instance_name,
            result.algorithm_name,
            result.execution_time.as_millis(),
            result.total_cost,
            result.best_known,
            result.solution_quality,
            result.solution.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(" ")
        ).expect("Unable to write to file");
    }

    println!("Results saved to {}", filename);
}

fn main() {
    benchmark(Solver::NearestNeighbor, vec![]);
    benchmark(Solver::GeneticAlgorithm, [100 as f64, 5 as f64, 0.01, 500 as f64].into());
    benchmark(Solver::SimulatedAnnealing, [1000.0, 0.96, 0.005, 66000 as f64].into());
    benchmark(Solver::AntColonySystem, [0.1, 2.0, 0.1, 0.9, 10.0, 1000.0].into());
    benchmark(Solver::RedBlackAntColonySystem, [1.0, 2.0, 0.1, 0.9, 20.0, 1000.0].into());
}