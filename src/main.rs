use ibn_battuta::algorithms::utils::Solver;
use ibn_battuta::algorithms::*;
use ibn_battuta::parser::TspBuilder;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};
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
    algorithms: &[Solver],
    params: &[Vec<f64>],
    num_threads: usize,
    _csv_file: Arc<Mutex<std::fs::File>>,
) {
    // println!("Starting parallel benchmarks with {} threads", num_threads);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    pool.install(|| {
        instances.par_iter().for_each(|instance| {
            // println!("Processing instance: {}", instance.path);
            algorithms.par_iter().enumerate().for_each(|(idx, algorithm)| {
                let params = &params[idx];
                // println!("> Benchmarking {} on instance: {}", algorithm, instance.path);
                let result = run_benchmark_multiple(instance, *algorithm, params.clone(), 3);
                // println!("< Finished benchmarking {} on instance: {}", algorithm, instance.path);

                // Write result to CSV file immediately
                // write_result_to_csv(&result, &csv_file);
                print_benchmark_result(&result);
            });
        });
    });
    // println!("Finished all parallel benchmarks");
}

fn _write_result_to_csv(result: &BenchmarkResult, csv_file: &Arc<Mutex<std::fs::File>>) {
    let mut file = csv_file.lock().unwrap();
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

fn run_benchmark_multiple(
    instance: &TspInstance,
    algorithm: Solver,
    params: Vec<f64>,
    num_runs: usize,
) -> BenchmarkResult {
    // println!("Starting {} runs for {} on instance {}", num_runs, algorithm, instance.path);
    let results: Vec<BenchmarkResult> = (0..num_runs)
        .into_par_iter()
        .map(|_i| {
            // println!("Benchmarking {} on instance {} run {} ", algorithm, instance.path, i);
            let tsp = Arc::new({
                match TspBuilder::parse_path(&instance.path) {
                    Ok(tsp) => tsp,
                    Err(e) => {
                        eprintln!("Error parsing TSP instance {} :{}", instance.path, e);
                        std::process::exit(1);
                    }
                }
            });
            let start = Instant::now();
            let mut solver = build_solver(instance.path.clone(), algorithm, &params);
            let solution = solver.solve();
            let duration = start.elapsed();

            let quality = (solution.total - instance.best_known) / instance.best_known * 100.0;
            // println!("Finished run {} for {} on instance {}", i, algorithm, instance.path);
            BenchmarkResult {
                instance_name: tsp.name().to_string(),
                algorithm_name: format!("{}", solver),
                execution_time: duration,
                total_cost: solution.total,
                best_known: instance.best_known,
                solution_quality: quality,
                solution: solution.tour,
            }
        })
        .collect();

    let best_result = results.iter()
        .min_by(|a, b| a.solution_quality.partial_cmp(&b.solution_quality).unwrap())
        .unwrap()
        .clone();

    let total_duration: Duration = results.iter().map(|r| r.execution_time).sum();
    let mut final_result = best_result;
    final_result.execution_time = total_duration / num_runs as u32;  // Average execution time
    // println!("Completed all runs for {} on instance {}", algorithm, instance.path);
    final_result
}

fn build_solver<'a>(instance: String, algorithm: Solver, params: &Vec<f64>) -> Box<dyn TspSolver + 'a> {
    let tsp = TspBuilder::parse_path(&instance).unwrap();
    match algorithm {
        Solver::GeneticAlgorithm => {
            let population_size = params[0] as usize;
            let elite_size = params[1] as usize;
            let crossover_rate = params[2];
            let mutation_rate = params[3];
            let max_generations = params[4] as usize;
            Box::new(GeneticAlgorithm::with_options(tsp, population_size,
                                                    elite_size,
                                                    crossover_rate,
                                                    mutation_rate,
                                                    max_generations, ))
        }

        Solver::GeneticAlgorithm2Opt => {
            let population_size = params[0] as usize;
            let elite_size = params[1] as usize;
            let crossover_rate = params[2];
            let mutation_rate = params[3];
            let max_generations = params[4] as usize;
            Box::new(GA2Opt::with_options(tsp, population_size,
                                          elite_size,
                                          crossover_rate,
                                          mutation_rate,
                                          max_generations, ))
        }

        Solver::NearestNeighbor => {
            Box::new(NearestNeighbor::new(tsp))
        }
        Solver::TwoOpt => {
            Box::new(TwoOpt::new(tsp))
        }
        Solver::LinKernighan => {
            let mut nn = NearestNeighbor::new(tsp.clone());
            let base_tour = nn.solve().tour;
            Box::new(LinKernighan::with_options(tsp, base_tour, true, 1000))
        }
        Solver::SimulatedAnnealing => {
            // let initial_temperature = params[0];
            // let cooling_rate = params[1];
            // let min_temperature = params[2];
            // let max_iterations = params[3] as usize;
            // let cycles_per_temperature = params[4] as usize;
            Box::new(SimulatedAnnealing::new(tsp))
        }
        Solver::SimulatedAnnealing2Opt => {
            // let initial_temperature = params[0];
            // let cooling_rate = params[1];
            // let min_temperature = params[2];
            // let max_iterations = params[3] as usize;
            // let cycles_per_temperature = params[4] as usize;
            Box::new(SA2Opt::new(tsp))
        }

        Solver::AntColonySystem => {
            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let q0 = params[3];
            let max_iterations = params[4] as usize;
            let candidate_list_size = params[5] as usize;
            let num_ants = 10;
            Box::new(AntColonySystem::with_options(tsp, alpha, beta, rho, q0, num_ants, max_iterations, candidate_list_size))
        }
        Solver::AntColonySystem2Opt => {
            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let q0 = params[3];
            let max_iterations = params[4] as usize;
            let candidate_list_size = params[5] as usize;
            let num_ants = 10;
            Box::new(ACS2Opt::with_options(tsp, alpha, beta, rho, q0, num_ants, max_iterations, candidate_list_size))
        }

        Solver::RedBlackAntColonySystem => {
            let alpha = params[0];
            let beta = params[1];
            let rho_red = params[2];
            let rho_black = params[3];
            let q0 = params[4];
            let num_ants = 10;
            let max_iterations = params[5] as usize;
            let candidate_list_size = params[6] as usize;

            Box::new(RedBlackACS::new(tsp, alpha, beta, rho_red, rho_black, q0,
                                      num_ants, max_iterations, candidate_list_size))
        }

        Solver::RedBlackAntColonySystem2Opt => {
            let alpha = params[0];
            let beta = params[1];
            let rho_red = params[2];
            let rho_black = params[3];
            let q0 = params[4];
            let num_ants = 10;
            let max_iterations = params[5] as usize;
            let candidate_list_size = params[6] as usize;

            Box::new(RBACS2Opt::with_options(tsp, alpha, beta, rho_red, rho_black, q0,
                                             num_ants, max_iterations, candidate_list_size))
        }

        Solver::AntSystem => {
            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let max_iterations = params[4] as usize;
            let num_ants = 20;
            Box::new(AntSystem::with_options(tsp, alpha, beta, rho, num_ants, max_iterations))
        }
        _ => unimplemented!(),
    }
}

fn benchmark(solvers: &[Solver], params: &[Vec<f64>], num_threads: usize) {
    // println!("Starting benchmark process");
    let instances_names = vec![
        ("eil51", 426.0),
        ("berlin52", 7542.0),
        ("st70", 675.0),
        ("pr76", 108159.0),
        ("eil76", 538.0),
        ("lin105", 14379.0),
        ("pr124", 59030.0),
        ("d198", 15780.0),
        ("a280", 2579.0),
        ("lin318", 42029.0),
        ("u574", 36905.0),
        ("rat575", 6773.0),
        ("p654", 34643.0),
        ("d657", 48912.0),
        ("rat783", 8806.0),
        ("pr1002", 259045.0),
        ("pcb1173", 56892.0),
        ("fl1577", 22249.0),
        // ("usa13509", 19982859.0),
        // ("d18512", 645238.0),
    ];

    let instances: Vec<TspInstance> = instances_names
        .iter()
        .map(|(name, best_known)| TspInstance {
            path: format!("data/tsplib/{}.tsp", name),
            best_known: *best_known,
        }).collect();

    // println!("Prepared {} instances for benchmarking", instances.len());

    let csv_file = Arc::new(Mutex::new(create_csv_file("Parallel-TSP-Benchmark.csv")));

    // Write CSV header
    {
        let mut file = csv_file.lock().unwrap();
        writeln!(file, "Instance,Algorithm,Time_ms,Length,Optimum,Gap,Solution").expect("Unable to write to file");
        println!("instance,algorithm,time_ms,length,optimum,gap,solution");
    }

    // println!("Starting parallel benchmarks");
    run_parallel_benchmarks(&instances, solvers, params, num_threads, csv_file.clone());

    // println!("Benchmarking complete. Results saved to Parallel-TSP-Benchmark.csv");
}

fn create_csv_file(filename: &str) -> std::fs::File {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .expect("Unable to create file")
}

#[allow(dead_code)]
fn print_benchmark_result(result: &BenchmarkResult) {
    println!(
        "{},{},{},{:.2},{:.2},{:.2},\"{}\"",
        result.instance_name,
        result.algorithm_name,
        result.execution_time.as_millis(),
        result.total_cost,
        result.best_known,
        result.solution_quality,
        result.solution.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(" ")
    );
}

#[allow(dead_code)]
fn save_results_to_csv(results: &[BenchmarkResult], filename: &str) {
    let mut file = File::create(filename).expect("Unable to create file");

    writeln!(file, "Instance,Algorithm,Time (ms),Found Tour Length,Best Known Length,Gap (%),Solution").expect("Unable to write to file");

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
}

fn main() {
    // println!("Starting TSP benchmark program");
    let solvers = vec![
        Solver::NearestNeighbor,
        Solver::TwoOpt,
        Solver::SimulatedAnnealing,
        Solver::SimulatedAnnealing2Opt,
        Solver::GeneticAlgorithm,
        Solver::GeneticAlgorithm2Opt,
        Solver::AntColonySystem,
        Solver::AntColonySystem2Opt,
        Solver::RedBlackAntColonySystem,
        Solver::RedBlackAntColonySystem2Opt,
        // Solver::AntSystem,
    ];

    let params = vec![
        vec![], // NN
        vec![], // NN+2-OPT
        vec![1000.0, 0.999, 0.0001, 1000.0, 100.0], // SA
        vec![1000.0, 0.999, 0.0001, 1000.0, 100.0], // SA-2OPT
        vec![100.0, 5.0, 0.7, 0.01, 500.0], // GA
        vec![100.0, 5.0, 0.7, 0.01, 500.0], // GA-2OPT
        vec![0.1, 2.0, 0.1, 0.9, 1000.0, 15.0], // ACS
        vec![0.1, 2.0, 0.1, 0.9, 1000.0, 15.0], // ACS-2OPT
        vec![0.1, 2.0, 0.1, 0.2, 0.9, 1000.0, 15.0], // RB-ACS
        vec![0.1, 2.0, 0.1, 0.2, 0.9, 1000.0, 15.0], // RB-ACS-2OPT
        // vec![0.1, 2.0, 0.1, 15.0, 1000.0], // AS
    ];

    let num_threads = 108;
    // println!("Configured {} solvers with {} threads", solvers.len(), num_threads);
    benchmark(&solvers, &params, num_threads);
    eprintln!("Benchmark program completed");
}