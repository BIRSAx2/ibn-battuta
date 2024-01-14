use std::path::Path;

use approx::assert_relative_eq;

use ibn_battuta::algorithms::{nearest_neighbor, SolverOptions};
use ibn_battuta::parser::*;
use ibn_battuta::Point;

#[test]
fn test_nearest_neighbor() {
    let path = Path::new("data/tsplib/berlin52.tsp");
    let tsp = TspBuilder::parse_path(path).unwrap();
    println!("{:?}", tsp);
    let options = SolverOptions::default();
    let mut node_coords = tsp.node_coords().iter().map(|(_, node)| node.clone()).collect::<Vec<Point>>();
    node_coords.sort_by(|a, b| a.id().cmp(&b.id()));
    let solution = nearest_neighbor::solve(&node_coords, &options);
    assert_relative_eq!(solution.total, 21653.847828728554);
    assert_eq!(solution.route().len(), tsp.dim());
}