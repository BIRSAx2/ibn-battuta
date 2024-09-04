use ibn_battuta::algorithms::{GeneticAlgorithm, TspSolver};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tspf::{Tsp, TspBuilder};

// Define a struct to hold TSP instance data
pub struct TspInstance {
    pub tsp: Tsp,
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
use std::sync::Arc;

fn run_parallel_benchmarks(
    instances: &[TspInstance],
    population_size: usize,
    tournament_size: usize,
    mutation_rate: f64,
    max_generations: usize,
) -> Vec<BenchmarkResult> {
    let instances = Arc::new(instances.to_vec());
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for instance in instances.iter() {
        let instance = instance.clone();
        let results = Arc::clone(&results);
        let instances = Arc::clone(&instances);
        let handle = thread::spawn(move || {
            let result = run_benchmark_multiple(&instance, population_size, tournament_size, mutation_rate, max_generations, 5);
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
    population_sizes: &[usize],
    tournament_sizes: &[usize],
    mutation_rates: &[f64],
    max_generations: &[usize],
) -> Vec<BenchmarkResult> {
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for &pop_size in population_sizes {
        for &tourn_size in tournament_sizes {
            for &mut_rate in mutation_rates {
                for &max_gen in max_generations {
                    let instance = instance.clone();
                    let results = Arc::clone(&results);
                    let handle = thread::spawn(move || {
                        println!("Running benchmark with pop_size: {}, tourn_size: {}, mut_rate: {}, max_gen: {}", pop_size, tourn_size, mut_rate, max_gen);
                        let result = run_benchmark_multiple(&instance, pop_size, tourn_size, mut_rate, max_gen, 5);
                        results.lock().unwrap().push(result);
                    });
                    handles.push(handle);
                }
            }
        }
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}


fn benchmark() {
    // Define TSP instances
    let instances = vec![
        TspInstance {
            tsp: TspBuilder::parse_path("data/tsplib/gr17.tsp").unwrap(),
            best_known: 2085.0,
        },
        TspInstance {
            tsp: TspBuilder::parse_path("data/tsplib/gr21.tsp").unwrap(),
            best_known: 2707.0,
        },
        TspInstance {
            tsp: TspBuilder::parse_path("data/tsplib/gr24.tsp").unwrap(),
            best_known: 1272.0,
        },
        TspInstance {
            tsp: TspBuilder::parse_path("data/tsplib/gr48.tsp").unwrap(),
            best_known: 5046.0,
        },
    ];

    // Benchmark on different instances in parallel
    println!("Benchmarking on different instances:");
    let instance_results = run_parallel_benchmarks(&instances, 100, 5, 0.01, 500);
    for result in &instance_results {
        print_benchmark_result(result);
    }

    // Save instance benchmark results to CSV
    save_results_to_csv(&instance_results, "ga_instance_benchmark.csv");

    // Grid search on gr48 instance
    println!("\nGrid search on gr48 instance:");
    let gr48 = &instances[3];
    let population_sizes = vec![100, 200];
    let tournament_sizes = vec![5, 10];
    let mutation_rates = vec![0.01, 0.02];
    let max_generations = vec![500, 1000];

    let grid_search_results = run_parallel_grid_search(gr48, &population_sizes, &tournament_sizes, &mutation_rates, &max_generations);

    let best_result = grid_search_results.iter().min_by(|a, b| a.solution_quality.partial_cmp(&b.solution_quality).unwrap()).unwrap();

    println!("Best configuration for gr48:");
    print_benchmark_result(best_result);

    // Save grid search results to CSV
    save_results_to_csv(&grid_search_results, "ga_grid_search.csv");
}

fn run_benchmark_multiple(
    instance: &TspInstance,
    population_size: usize,
    tournament_size: usize,
    mutation_rate: f64,
    max_generations: usize,
    num_runs: usize,
) -> BenchmarkResult {
    let mut best_result = None;
    let mut best_quality = f64::INFINITY;
    let mut total_duration = Duration::new(0, 0);

    for _ in 0..num_runs {
        let start = Instant::now();
        let mut solver = GeneticAlgorithm::with_options(
            &instance.tsp,
            population_size,
            tournament_size,
            mutation_rate,
            max_generations,
        );
        let solution = solver.solve();
        let duration = start.elapsed();
        total_duration += duration;

        let quality = (solution.total - instance.best_known) / instance.best_known * 100.0;
        if quality < best_quality {
            best_quality = quality;
            best_result = Some(BenchmarkResult {
                instance_name: instance.tsp.name().to_string(),
                algorithm_name: format!(
                    "GA (pop: {}, tourn: {}, mut: {:.3}, gen: {})",
                    population_size, tournament_size, mutation_rate, max_generations
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
    let mut solver = GeneticAlgorithm::with_options(
        &instance.tsp,
        population_size,
        tournament_size,
        mutation_rate,
        max_generations,
    );
    let solution = solver.solve();
    let duration = start.elapsed();

    BenchmarkResult {
        instance_name: instance.tsp.name().to_string(),
        algorithm_name: format!(
            "GA (pop: {}, tourn: {}, mut: {:.3}, gen: {})",
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
    writeln!(file, "Instance;Algorithm;Time (ms);Found Tour Length;Best Known Length;Gap (%);Solution").expect("Unable to write to file");

    // Write data
    for result in results {
        writeln!(
            file,
            "{};{};{};{:.2};{:.2};{:.2};\"{}\"",
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
    benchmark();
}