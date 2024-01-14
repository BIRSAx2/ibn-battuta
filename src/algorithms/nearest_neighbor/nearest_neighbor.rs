use std::cmp::Ordering;
use std::collections::HashMap;

use crate::algorithms::{Solution, SolverOptions};
use crate::algorithms::nearest_neighbor::kd_tree::{KDNode, KDTree};
use crate::utils::point::Point;

/// A matrix of points in a k-dimensional space.
pub type PointMatrix = Vec<Vec<f64>>;

/// A type alias for a boxed KDNode, representing a subtree in the KDTree.
pub type KDSubTree = Option<Box<KDNode>>;


pub fn solve(cities: &[Point], options: &SolverOptions) -> Solution {
    let search_tree = from_cities(&cities);
    let n_nearest = options.n_nearest;

    let cities_table: HashMap<usize, Point> = cities.iter().map(|c| (c.id(), c.clone())).collect();
    let mut path: Vec<usize> = cities.iter().map(|c| c.id()).collect();

    // run optimization round
    for i in 0..(path.len() - 1) {
        let id1 = path[i];
        let city1 = cities_table[&id1].clone();

        let frontier = search_tree.nearest(&city1, n_nearest);

        let id2 = path[i + 1];
        let current_distance = city1.distance(&cities_table[&id2]);

        let search_result = frontier.nearest();
        if search_result.is_empty() {
            if options.verbose {
                println!("No nearest for city: #{:?}", id1);
            }

            continue;
        }

        let closest_item = search_result.first().unwrap();
        let next_distance = closest_item.distance;

        if next_distance < current_distance {
            let nearest_city_id = closest_item.point.id();
            if let Some(nearest_pos) = path.iter().position(|&x| x == nearest_city_id) {
                path.swap(i + 1, nearest_pos);
            }
        }
    }

    let tour = Solution::new(&path, cities);
    tour
}


/// Represents the result of a nearest neighbor search.
#[derive(Debug, Clone)]
pub struct NearestResult {
    /// The target point for the nearest neighbor search.
    pub target: Point,
    /// The current closest point found during the search.
    pub point: Point,
    /// The distance between the target and the current closest point.
    pub distance: f64,
    /// The maximum number of items to keep in the result.
    pub items_to_keep: usize,
    /// A vector of additional nearest results.
    results: Vec<NearestResultItem>,
}

impl NearestResult {
    /// Creates a new `NearestResult`.
    ///
    /// # Arguments
    ///
    /// * `point` - The target point for the nearest neighbor search.
    /// * `distance` - The initial distance to set for the search.
    /// * `n` - The maximum number of items to keep in the result.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{NearestResult, Point};
    ///
    /// let point = Point::new_with_coords(&vec![1.0, 2.0]);
    /// let nearest_result = NearestResult::new(point.clone(), f64::INFINITY, 5);
    /// ```
    pub fn new(point: Point, distance: f64, n: usize) -> Self {
        let results = Vec::with_capacity(n);

        NearestResult {
            target: point.clone(),
            point,
            distance,
            items_to_keep: n,
            results,
        }
    }

    /// Adds a new point to the result if it is closer than the current farthest point in the result.
    ///
    /// # Arguments
    ///
    /// * `pt` - The point to add.
    /// * `new_distance` - The distance from the target to the new point.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{NearestResult, Point};
    ///
    /// let point = Point::new_with_coords(&vec![1.0, 2.0]);
    /// let mut nearest_result = NearestResult::new(point.clone(), f64::INFINITY, 5);
    ///
    /// let new_point = Point::new_with_coords(&vec![3.0, 4.0]);
    /// nearest_result.add(new_point, 5.0);
    /// ```
    pub fn add(&mut self, pt: Point, new_distance: f64) {
        if self.items_to_keep == 0 || pt.id() == self.target.id() {
            return;
        }

        if new_distance < self.closest_distance() {
            self.distance = new_distance;
            self.point = pt.clone();
        }

        // we only keep the best results
        if new_distance < self.farthest_distance() {
            // if stack is full, then remove the weakest result
            if self.results.len() >= self.items_to_keep {
                self.results.pop();
            }

            let new_result = NearestResultItem::new(pt, new_distance);
            self.results.push(new_result);
            self.results.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    /// Returns a vector of additional nearest results.
    ///
    /// # Returns
    ///
    /// A vector containing additional nearest results.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{NearestResult, Point};
    ///
    /// let point = Point::new_with_coords(&vec![1.0, 2.0]);
    /// let nearest_result = NearestResult::new(point.clone(), f64::INFINITY, 5);
    ///
    /// let additional_results = nearest_result.nearest();
    /// ```
    pub fn nearest(&self) -> Vec<&NearestResultItem> {
        self.results.iter().collect::<Vec<&NearestResultItem>>()
    }

    /// Returns the distance to the closest point.
    ///
    /// # Returns
    ///
    /// The distance to the closest point found during the search.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{NearestResult, Point};
    ///
    /// let point = Point::new_with_coords(&vec![1.0, 2.0]);
    /// let nearest_result = NearestResult::new(point.clone(), 5.0, 5);
    ///
    /// let closest_distance = nearest_result.closest_distance();
    /// ```
    pub fn closest_distance(&self) -> f64 {
        self.distance
    }

    /// Returns the distance to the farthest point in the result.
    ///
    /// # Returns
    ///
    /// The distance to the farthest point in the result, or `f64::MAX` if the result is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{NearestResult, Point};
    ///
    /// let point = Point::new_with_coords(&vec![1.0, 2.0]);
    /// let nearest_result = NearestResult::new(point.clone(), 5.0, 5);
    ///
    /// let farthest_distance = nearest_result.farthest_distance();
    /// ```
    pub fn farthest_distance(&self) -> f64 {
        self.results.last().map(|x| x.distance).unwrap_or(f64::MAX)
    }
}

/// Represents an item in a `NearestResult`, containing a point and its distance from the target.
#[derive(Debug, Clone)]
pub struct NearestResultItem {
    /// The distance from the target to the point.
    pub distance: f64,
    /// The point in the result.
    pub point: Point,
}

impl NearestResultItem {
    /// Creates a new `NearestResultItem`.
    ///
    /// # Arguments
    ///
    /// * `point` - The point in the result.
    /// * `distance` - The distance from the target to the point.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use ibn_battuta::algorithms::nearest_neighbor::NearestResultItem;
    /// use ibn_battuta::Point;
    /// let point = Point::new_with_coords(&vec![1.0, 2.0]);
    /// let nearest_result_item = NearestResultItem::new(point.clone(), 5.0);
    /// ```
    pub fn new(point: Point, distance: f64) -> Self {
        NearestResultItem { point, distance }
    }
}

impl PartialOrd for NearestResultItem {
    fn partial_cmp(&self, other: &NearestResultItem) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for NearestResultItem {
    fn eq(&self, other: &NearestResultItem) -> bool {
        self.distance == other.distance
    }
}


/// Builds a collection of `Point` instances from a `PointMatrix`.
///
/// # Arguments
///
/// * `rows` - A reference to a vector of vectors representing the `PointMatrix`.
///
/// # Returns
///
/// A vector of `Point` instances.
///
/// # Example
///
/// ```rust
///
/// use ibn_battuta::algorithms::nearest_neighbor::build_points;
/// let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
/// let points = build_points(&matrix);
/// ```
pub fn build_points(rows: &[Vec<f64>]) -> Vec<Point> {
    let mut points = vec![];

    for (i, coords) in rows.iter().enumerate() {
        points.push(Point::new_with_id(i, coords));
    }
    points
}

/// Builds a KDTree from a collection of points.
///
/// # Arguments
///
/// * `points` - A slice of `Point` instances.
///
/// # Returns
///
/// A `KDTree` constructed from the provided points.
///
/// # Example
///
/// ```rust
///
/// use ibn_battuta::algorithms::nearest_neighbor::from_cities;
/// use ibn_battuta::Point;
/// let points = vec![
///     Point::new_with_coords(&vec![1.0, 2.0]),
///     Point::new_with_coords(&vec![3.0, 4.0]),
///     Point::new_with_coords(&vec![5.0, 6.0]),
/// ];
///
/// let tree = from_cities(&points);
/// ```
pub fn from_cities(points: &[Point]) -> KDTree {
    let mut tree = KDTree::empty();

    if points.is_empty() {
        return tree;
    };

    let n_points = points.len();
    let tree_points = points.to_vec();
    if let Some(root) = build_subtree(tree_points, 0) {
        tree.size = n_points;
        tree.root = Some(root);
    }

    tree
}

/// Builds a subtree of a KDTree from a collection of points.
///
/// # Arguments
///
/// * `points` - A vector of `Point` instances.
/// * `depth` - The depth of the current node in the KDTree.
///
/// # Returns
///
/// A `KDSubTree` representing the root of the subtree constructed from the provided points.

fn build_subtree(points: Vec<Point>, depth: usize) -> KDSubTree {
    if points.is_empty() {
        return None;
    }

    if points.len() == 1 {
        let leaf_node = KDNode::leaf(points[0].clone(), depth);
        return Some(Box::new(leaf_node));
    }

    let k = points[0].dimensionality();
    let (pivot_pt, left_points, right_points) = partition_points(points, depth, k);
    let root = KDNode::from_subtrees(
        pivot_pt,
        depth,
        build_subtree(left_points, depth + 1),
        build_subtree(right_points, depth + 1),
    );

    Some(Box::new(root))
}

/// Partitions a collection of points into a pivot point, left points, and right points.
///
/// # Arguments
///
/// * `points` - A vector of `Point` instances.
/// * `depth` - The depth of the current node in the KDTree.
/// * `k` - The dimensionality of the points.
///
/// # Returns
///
/// A tuple containing the pivot point, left points, and right points.

pub fn partition_points(
    points: Vec<Point>,
    depth: usize,
    k: usize,
) -> (Point, Vec<Point>, Vec<Point>) {
    let mut sorted_points = points.clone();

    if sorted_points.len() == 1 {
        let pivot_pt = sorted_points[0].clone();
        return (pivot_pt, vec![], vec![]);
    }

    let coord = depth % k;
    sorted_points.sort_by(|a, b| a.cmp_by_coord(&b, coord).unwrap());

    let pivot_idx = sorted_points.len() / 2;
    let pivot_pt = sorted_points[pivot_idx].clone();

    (
        pivot_pt,
        sorted_points[0..pivot_idx].to_vec(),
        sorted_points[(pivot_idx + 1)..].to_vec(),
    )
}

