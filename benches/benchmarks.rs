use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ibn_battuta::algorithms::*;
use std::time::Duration;
use tspf::{Tsp, TspBuilder};

// Define a struct to hold TSP instance data
struct TspInstance {
    tsp: Tsp,
    best_known: f64,
}

// Define TSP instances
fn get_tsp_instances() -> Vec<TspInstance> {
    vec![
        TspInstance {
            tsp: TspBuilder::parse_path("data/tsplib/gr17.tsp").unwrap(),
            best_known: 19.0,
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
            best_known: 937.0,
        },
        TspInstance {
            tsp: TspBuilder::parse_path("data/tsplib/gr96.tsp").unwrap(),
            best_known: 6942.0,
        },
    ]
}

// Benchmark function for comparing algorithms on the same instance
fn bench_algorithms_same_instance(c: &mut Criterion) {
    let instances = get_tsp_instances();
    let mut group = c.benchmark_group("TSP Algorithms");

    for instance in instances.iter() {
        let tsp = &instance.tsp;


        group.bench_with_input(BenchmarkId::new("Nearest Neighbor", &tsp.name()), &tsp,
                               |b, tsp| b.iter(|| {
                                   let mut solver = NearestNeighbor::new(tsp);
                                   black_box(solver.solve())
                               }));

        // Add more algorithms as needed
    }

    group.finish();
}

// Benchmark function for comparing instances using the same algorithm

fn bench_instances_same_algorithm(c: &mut Criterion) {
    let instances = get_tsp_instances();
    let mut group = c.benchmark_group("TSP Instances");

    let algorithms: Vec<(&str, Box<dyn Fn(&Tsp) -> Solution>)> = vec![
        ("Nearest Neighbor", Box::new(|tsp: &Tsp| {
            let mut solver = NearestNeighbor::new(tsp);
            solver.solve()
        })),
        // Add more algorithms as needed
    ];

    for (algo_name, algo_fn) in algorithms.iter() {
        for instance in instances.iter() {
            let tsp = &instance.tsp;
            group.bench_with_input(BenchmarkId::new(*algo_name, &tsp.name()), &tsp,
                                   |b, tsp| b.iter(|| black_box(algo_fn(tsp))));
        }
    }

    group.finish();
}

criterion_group! {
    name = tsp_benchmarks;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = bench_algorithms_same_instance, bench_instances_same_algorithm
}

criterion_main!(tsp_benchmarks);