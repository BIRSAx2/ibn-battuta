use ibn_battuta::algorithms::utils::{Solver, SolverSupport};
use ibn_battuta::experimental::{
    ACS2Opt, AntColonySystem, AntSystem, GA2Opt, GeneticAlgorithm, LinKernighan, RBACS2Opt,
    RedBlackACS, SA2Opt, SimulatedAnnealing,
};
use ibn_battuta::{NearestNeighbor, Tsp, TspBuilder, TspSolver, TwoOpt};
use rayon::prelude::*;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const BENCHMARK_BASE_SEED: u64 = 42;
const DEFAULT_FOCUSED_RUNS: usize = 5;
const DEFAULT_SMALL_RUNS: usize = 3;
const DEFAULT_FULL_RUNS: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkProfile {
    Focused,
    Small,
    Full,
}

#[derive(Clone, Debug)]
struct BenchmarkConfig {
    profile: BenchmarkProfile,
    num_runs: usize,
    num_threads: usize,
    csv_path: String,
}

impl BenchmarkConfig {
    fn from_env() -> Self {
        let profile = match env::var("IBN_BATTUTA_BENCH_PROFILE")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("full") => BenchmarkProfile::Full,
            Some("focused") => BenchmarkProfile::Focused,
            _ => BenchmarkProfile::Small,
        };

        let default_runs = match profile {
            BenchmarkProfile::Focused => DEFAULT_FOCUSED_RUNS,
            BenchmarkProfile::Small => DEFAULT_SMALL_RUNS,
            BenchmarkProfile::Full => DEFAULT_FULL_RUNS,
        };
        let num_runs = env_usize("IBN_BATTUTA_BENCH_RUNS").unwrap_or(default_runs);
        let num_threads =
            env_usize("IBN_BATTUTA_BENCH_THREADS").unwrap_or_else(default_thread_count);
        let csv_path = env::var("IBN_BATTUTA_BENCH_CSV")
            .unwrap_or_else(|_| "Parallel-TSP-Benchmark.csv".to_string());

        BenchmarkConfig {
            profile,
            num_runs,
            num_threads,
            csv_path,
        }
    }
}

fn env_usize(key: &str) -> Option<usize> {
    env::var(key).ok()?.trim().parse().ok()
}

fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn env_f64_list(key: &str) -> Option<Vec<f64>> {
    env::var(key).ok().and_then(|raw| {
        let values = raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        (!values.is_empty()).then_some(values)
    })
}

fn selected_solvers(defaults: &[(Solver, Vec<f64>)]) -> Vec<(Solver, Vec<f64>)> {
    let Some(raw) = env::var("IBN_BATTUTA_BENCH_SOLVERS").ok() else {
        return defaults.to_vec();
    };

    let selected: Vec<Solver> = raw
        .split(',')
        .map(str::trim)
        .filter_map(parse_solver_name)
        .collect();

    if selected.is_empty() {
        return defaults.to_vec();
    }

    defaults
        .iter()
        .filter(|(solver, _)| selected.contains(solver))
        .cloned()
        .collect()
}

fn parse_solver_name(name: &str) -> Option<Solver> {
    match name.trim().to_ascii_lowercase().as_str() {
        "nn" | "nearestneighbor" => Some(Solver::NearestNeighbor),
        "2opt" | "twoopt" => Some(Solver::TwoOpt),
        "sa" | "simulatedannealing" => Some(Solver::SimulatedAnnealing),
        "sa2opt" => Some(Solver::SimulatedAnnealing2Opt),
        "ga" | "geneticalgorithm" => Some(Solver::GeneticAlgorithm),
        "ga2opt" => Some(Solver::GeneticAlgorithm2Opt),
        "acs" | "antcolonysystem" => Some(Solver::AntColonySystem),
        "acs2opt" => Some(Solver::AntColonySystem2Opt),
        "rbacs" | "redblackacs" | "redblackantcolonysystem" => {
            Some(Solver::RedBlackAntColonySystem)
        }
        "rbacs2opt" | "redblackantcolonysystem2opt" => Some(Solver::RedBlackAntColonySystem2Opt),
        "as" | "antsystem" => Some(Solver::AntSystem),
        "lk" | "linkernighan" => Some(Solver::LinKernighan),
        _ => None,
    }
}

fn instance_catalog() -> Vec<(&'static str, f64)> {
    vec![
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
        ("d1655", 62128.0),
        ("d2103", 80450.0),
        ("u2319", 234256.0),
        ("rl5915", 565530.0),
    ]
}

fn instances_for_profile(profile: BenchmarkProfile) -> Vec<TspInstance> {
    let names = match profile {
        BenchmarkProfile::Focused => vec![("lin105", 14379.0)],
        BenchmarkProfile::Small => vec![("eil51", 426.0), ("berlin52", 7542.0), ("st70", 675.0)],
        BenchmarkProfile::Full => instance_catalog(),
    };

    names
        .into_iter()
        .map(|(name, best_known)| TspInstance {
            path: format!("data/tsplib/{}.tsp", name),
            best_known,
        })
        .collect()
}

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
    pub support_level: SolverSupport,
    pub seed: u64,
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
    config: &BenchmarkConfig,
    csv_file: Arc<Mutex<std::fs::File>>,
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads)
        .build()
        .unwrap();

    pool.install(|| {
        instances.par_iter().for_each(|instance| {
            let tsp = match TspBuilder::parse_path(&instance.path) {
                Ok(tsp) => tsp,
                Err(error) => {
                    eprintln!("Error parsing TSP instance {}: {}", instance.path, error);
                    return;
                }
            };

            algorithms
                .par_iter()
                .enumerate()
                .for_each(|(idx, algorithm)| {
                    let params = &params[idx];
                    let result =
                        run_benchmark_multiple(&tsp, instance, *algorithm, params, config.num_runs);
                    _write_result_to_csv(&result, &csv_file);
                    print_benchmark_result(&result);
                });
        });
    });
}

fn _write_result_to_csv(result: &BenchmarkResult, csv_file: &Arc<Mutex<std::fs::File>>) {
    let mut file = csv_file.lock().unwrap();
    writeln!(
        file,
        "{},{},{},{},{},{:.2},{:.2},{:.2},\"{}\"",
        result.instance_name,
        result.algorithm_name,
        result.support_level,
        result.seed,
        result.execution_time.as_millis(),
        result.total_cost,
        result.best_known,
        result.solution_quality,
        result
            .solution
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(" ")
    )
    .expect("Unable to write to file");
}

fn run_benchmark_multiple(
    tsp: &Tsp,
    instance: &TspInstance,
    algorithm: Solver,
    params: &[f64],
    num_runs: usize,
) -> BenchmarkResult {
    let results: Vec<BenchmarkResult> = (0..num_runs)
        .into_par_iter()
        .map(|run_idx| {
            let seed = BENCHMARK_BASE_SEED.wrapping_add(run_idx as u64);
            let start = Instant::now();
            let mut solver = build_solver(tsp.clone(), algorithm, params, seed);
            let solution = solver.solve();
            let duration = start.elapsed();

            let quality = (solution.length - instance.best_known) / instance.best_known * 100.0;
            BenchmarkResult {
                instance_name: tsp.name().to_string(),
                algorithm_name: format!("{}", solver),
                support_level: algorithm.support(),
                seed,
                execution_time: duration,
                total_cost: solution.length,
                best_known: instance.best_known,
                solution_quality: quality,
                solution: solution.tour,
            }
        })
        .collect();

    let best_result = results
        .iter()
        .min_by(|a, b| a.solution_quality.partial_cmp(&b.solution_quality).unwrap())
        .unwrap()
        .clone();

    let total_duration: Duration = results.iter().map(|r| r.execution_time).sum();
    let mut final_result = best_result;
    final_result.execution_time = total_duration / num_runs as u32; // Average execution time
    final_result
}

fn build_solver<'a>(
    tsp: Tsp,
    algorithm: Solver,
    params: &[f64],
    seed: u64,
) -> Box<dyn TspSolver + 'a> {
    match algorithm {
        Solver::GeneticAlgorithm => {
            let population_size = params[0] as usize;
            let elite_size = params[1] as usize;
            let crossover_rate = params[2];
            let mutation_rate = params[3];
            let max_generations = params[4] as usize;
            Box::new(GeneticAlgorithm::with_options_and_seed(
                tsp,
                population_size,
                elite_size,
                crossover_rate,
                mutation_rate,
                max_generations,
                seed,
            ))
        }

        Solver::GeneticAlgorithm2Opt => {
            let population_size = params[0] as usize;
            let elite_size = params[1] as usize;
            let crossover_rate = params[2];
            let mutation_rate = params[3];
            let max_generations = params[4] as usize;
            Box::new(GA2Opt::with_options_and_seed(
                tsp,
                population_size,
                elite_size,
                crossover_rate,
                mutation_rate,
                max_generations,
                seed,
            ))
        }

        Solver::NearestNeighbor => Box::new(NearestNeighbor::new(tsp)),
        Solver::TwoOpt => Box::new(TwoOpt::new(tsp)),
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
            Box::new(SimulatedAnnealing::with_seed(tsp, seed))
        }
        Solver::SimulatedAnnealing2Opt => {
            // let initial_temperature = params[0];
            // let cooling_rate = params[1];
            // let min_temperature = params[2];
            // let max_iterations = params[3] as usize;
            // let cycles_per_temperature = params[4] as usize;
            Box::new(SA2Opt::new_with_seed(tsp, seed))
        }

        Solver::AntColonySystem => {
            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let q0 = params[3];
            let max_iterations = params[4] as usize;
            let candidate_list_size = params[5] as usize;
            let num_ants = env_usize("IBN_BATTUTA_ACS_NUM_ANTS").unwrap_or(10);
            Box::new(AntColonySystem::with_options_and_seed(
                tsp,
                alpha,
                beta,
                rho,
                q0,
                num_ants,
                max_iterations,
                candidate_list_size,
                seed,
            ))
        }
        Solver::AntColonySystem2Opt => {
            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let q0 = params[3];
            let max_iterations = params[4] as usize;
            let candidate_list_size = params[5] as usize;
            let num_ants = env_usize("IBN_BATTUTA_ACS_NUM_ANTS").unwrap_or(10);
            Box::new(ACS2Opt::with_options_and_seed(
                tsp,
                alpha,
                beta,
                rho,
                q0,
                num_ants,
                max_iterations,
                candidate_list_size,
                seed,
            ))
        }

        Solver::RedBlackAntColonySystem => {
            let alpha = params[0];
            let beta = params[1];
            let rho_red = params[2];
            let rho_black = params[3];
            let q0 = params[4];
            let num_ants = env_usize("IBN_BATTUTA_RBACS_NUM_ANTS").unwrap_or(10);
            let max_iterations = params[5] as usize;
            let candidate_list_size = params[6] as usize;

            Box::new(RedBlackACS::new_with_seed(
                tsp,
                alpha,
                beta,
                rho_red,
                rho_black,
                q0,
                num_ants,
                max_iterations,
                candidate_list_size,
                seed,
            ))
        }

        Solver::RedBlackAntColonySystem2Opt => {
            let alpha = params[0];
            let beta = params[1];
            let rho_red = params[2];
            let rho_black = params[3];
            let q0 = params[4];
            let num_ants = env_usize("IBN_BATTUTA_RBACS_NUM_ANTS").unwrap_or(10);
            let max_iterations = params[5] as usize;
            let candidate_list_size = params[6] as usize;

            Box::new(RBACS2Opt::with_options_and_seed(
                tsp,
                alpha,
                beta,
                rho_red,
                rho_black,
                q0,
                num_ants,
                max_iterations,
                candidate_list_size,
                seed,
            ))
        }

        Solver::AntSystem => {
            let alpha = params[0];
            let beta = params[1];
            let rho = params[2];
            let max_iterations = params[4] as usize;
            let num_ants = 20;
            Box::new(AntSystem::with_options_and_seed(
                tsp,
                alpha,
                beta,
                rho,
                num_ants,
                max_iterations,
                seed,
            ))
        }
        _ => unimplemented!(),
    }
}

fn benchmark(solvers: &[Solver], params: &[Vec<f64>], config: &BenchmarkConfig) {
    let instances = instances_for_profile(config.profile);
    let csv_file = Arc::new(Mutex::new(create_csv_file(&config.csv_path)));

    // Write CSV header
    {
        let mut file = csv_file.lock().unwrap();
        writeln!(
            file,
            "Instance,Algorithm,Support,Seed,Time_ms,Length,Optimum,Gap,Solution"
        )
        .expect("Unable to write to file");
        println!("instance,algorithm,support,seed,time_ms,length,optimum,gap,solution");
        eprintln!(
            "benchmark profile={:?} runs={} threads={} instances={} csv={}",
            config.profile,
            config.num_runs,
            config.num_threads,
            instances.len(),
            config.csv_path
        );
    }

    run_parallel_benchmarks(&instances, solvers, params, config, csv_file.clone());
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
        "{},{},{},{},{},{:.2},{:.2},{:.2},\"{}\"",
        result.instance_name,
        result.algorithm_name,
        result.support_level,
        result.seed,
        result.execution_time.as_millis(),
        result.total_cost,
        result.best_known,
        result.solution_quality,
        result
            .solution
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(" ")
    );
}

#[allow(dead_code)]
fn save_results_to_csv(results: &[BenchmarkResult], filename: &str) {
    let mut file = File::create(filename).expect("Unable to create file");

    writeln!(
        file,
        "Instance,Algorithm,Support,Seed,Time (ms),Found Tour Length,Best Known Length,Gap (%),Solution"
    )
    .expect("Unable to write to file");

    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{:.2},{:.2},{:.2},\"{}\"",
            result.instance_name,
            result.algorithm_name,
            result.support_level,
            result.seed,
            result.execution_time.as_millis(),
            result.total_cost,
            result.best_known,
            result.solution_quality,
            result
                .solution
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<String>>()
                .join(" ")
        )
        .expect("Unable to write to file");
    }
}

fn main() {
    let mut solver_configs = vec![
        (Solver::NearestNeighbor, vec![]),
        (Solver::TwoOpt, vec![]),
        (
            Solver::SimulatedAnnealing,
            vec![1000.0, 0.999, 0.0001, 1000.0, 100.0],
        ),
        (
            Solver::SimulatedAnnealing2Opt,
            vec![1000.0, 0.999, 0.0001, 1000.0, 100.0],
        ),
        (Solver::GeneticAlgorithm, vec![100.0, 5.0, 0.7, 0.01, 500.0]),
        (
            Solver::GeneticAlgorithm2Opt,
            vec![100.0, 5.0, 0.7, 0.01, 500.0],
        ),
        (
            Solver::AntColonySystem,
            vec![0.1, 2.0, 0.1, 0.95, 1000.0, 20.0],
        ),
        (
            Solver::AntColonySystem2Opt,
            vec![0.1, 2.0, 0.1, 0.95, 1000.0, 20.0],
        ),
        (
            Solver::RedBlackAntColonySystem,
            vec![0.1, 2.0, 0.1, 0.2, 0.95, 1000.0, 20.0],
        ),
        (
            Solver::RedBlackAntColonySystem2Opt,
            vec![0.1, 2.0, 0.1, 0.2, 0.95, 1000.0, 20.0],
        ),
        // (Solver::AntSystem, vec![0.1, 2.0, 0.1, 15.0, 1000.0]),
    ];
    if let Some(params) = env_f64_list("IBN_BATTUTA_ACS_PARAMS") {
        for (solver, solver_params) in &mut solver_configs {
            if matches!(
                solver,
                Solver::AntColonySystem | Solver::AntColonySystem2Opt
            ) {
                *solver_params = params.clone();
            }
        }
    }
    if let Some(params) = env_f64_list("IBN_BATTUTA_RBACS_PARAMS") {
        for (solver, solver_params) in &mut solver_configs {
            if matches!(
                solver,
                Solver::RedBlackAntColonySystem | Solver::RedBlackAntColonySystem2Opt
            ) {
                *solver_params = params.clone();
            }
        }
    }

    let solver_configs = selected_solvers(&solver_configs);
    let (solvers, params): (Vec<_>, Vec<_>) = solver_configs.into_iter().unzip();

    let config = BenchmarkConfig::from_env();
    benchmark(&solvers, &params, &config);
    eprintln!("Benchmark program completed");
}
