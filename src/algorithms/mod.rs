use std::collections::HashMap;
use std::str::FromStr;

use crate::Point;

pub mod nearest_neighbor;


/// Represents different algorithms for solving the Traveling Salesman Problem (TSP).
#[derive(Clone, Debug, PartialEq)]
pub enum Solvers {
    NearestNeighbor,
    Unspecified,
}

impl Solvers {
    /// Returns a vector containing variant names for the Solvers enum.

    pub fn variants() -> Vec<&'static str> {
        vec![
            "nearest_neighbor",
            "nn",
        ]
    }
}

impl FromStr for Solvers {
    type Err = &'static str;

    ///  /// Attempts to convert a string to a Solver enum variant.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nn" | "nearest_neighbor" => Ok(Solvers::NearestNeighbor),
            _ => Err("unknown solver"),
        }
    }
}


/// Options for configuring a TSP solver.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    /// If true, enables verbose output.
    pub verbose: bool,
    /// The number of nearest neighbors to consider.
    pub n_nearest: usize,
    /// If true, displays progress information.
    pub show_progress: bool,
}


impl SolverOptions {
    /// Returns the default options for a TSP solver.
    pub fn default() -> Self {
        SolverOptions {
            verbose: false,
            n_nearest: 3,
            show_progress: true,
        }
    }
}


/// A mapping from city IDs to their corresponding Point in a TSP.
pub type CityTable = HashMap<usize, Point>;

/// Calculates the total distance of a given route through a set of cities.
///
/// # Arguments
///
/// * `cities` - The vector of Point representing cities.
/// * `route` - The vector of usize representing the order of cities in the route.
///
/// # Returns
///
/// The total distance of the route.
pub fn total_distance(cities: &[Point], route: &[usize]) -> f64 {
    let mut total = 0.0;
    let last_idx = route.len() - 1;

    let cities_table = city_table_from_vec(cities);
    for i in 0..last_idx {
        let distance = cities_table[&route[i]].distance(&cities_table[&route[i + 1]]);
        total += distance
    }

    total += cities_table[&route[last_idx]].distance(&cities_table[&route[0]]);

    total
}

/// Creates a CityTable mapping from city IDs to their corresponding Point.
///
/// # Arguments
///
/// * `cities` - The vector of Point representing cities.
///
/// # Returns
///
/// A CityTable mapping city IDs to their corresponding Point.
pub fn city_table_from_vec(cities: &[Point]) -> CityTable {
    let table: CityTable = cities.iter().map(|c| (c.id(), c.clone())).collect();

    return table;
}

/// Represents a solution to the Traveling Salesman Problem (TSP).
#[derive(Debug)]
pub struct Solution {
    /// The total distance of the TSP route.
    pub total: f64,
    /// The order of city IDs in the TSP route.
    route: Vec<usize>,
    /// The vector of Point representing cities in the TSP.
    cities: Vec<Point>,
    /// A mapping from city IDs to their index in the cities vector.
    cities_idx: HashMap<usize, usize>,
}

impl Solution {
    /// Creates a new TSP solution with the given route and cities.
    ///
    /// # Arguments
    ///
    /// * `route` - The order of city IDs in the TSP route.
    /// * `cities` - The vector of Point representing cities in the TSP.
    ///
    /// # Returns
    ///
    /// A new Solution instance.
    pub fn new(route: &[usize], cities: &[Point]) -> Self {
        let idx: HashMap<usize, usize> =
            cities.iter().enumerate().map(|(i, c)| (c.id(), i)).collect();

        let mut solution = Solution {
            total: 0.0,
            route: route.to_vec(),
            cities: cities.to_vec(),
            cities_idx: idx,
        };

        solution.update_total();

        solution
    }

    /// Returns the number of cities in the TSP route.
    pub fn len(&self) -> usize {
        self.route.len()
    }

    /// Returns a reference to the TSP route.
    pub fn route(&self) -> &[usize] {
        self.route[..].as_ref()
    }

    /// Returns a reference to the vector of Point representing cities in the TSP.
    pub fn cities(&self) -> &[Point] {
        &self.cities[..]
    }

    /// Returns the Point for a given city ID in the TSP route.
    ///
    /// # Arguments
    ///
    /// * `city_id` - The ID of the city to retrieve.
    ///
    /// # Returns
    ///
    /// A reference to the Point representing the city.
    pub fn get_by_city_id(&self, city_id: usize) -> Option<&Point> {
        if let Some(vec_pos) = self.cities_idx.get(&city_id) {
            self.cities.get(*vec_pos)
        } else {
            None
        }
    }

    /// Updates the total distance of the TSP solution based on its current route.
    pub fn update_total(&mut self) {
        self.total = total_distance(self.cities(), self.route());
    }
}