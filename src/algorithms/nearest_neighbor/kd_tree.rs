// TODO: This works only with euclidean distance, need to make it generic

//! # KDTree
//!
//! This module provides a simple implementation of a KDTree (K-Dimensional Tree), a space-partitioning data structure
//! used for organizing points in a k-dimensional space. It supports operations such as insertion and nearest neighbor
//! search efficiently.
//!
//! ## Example
//!
//! ```rust
//! use ibn_battuta::algorithms::nearest_neighbor::kd_tree::{KDTree};
//! use ibn_battuta::Point;
//!
//! // Create a KDTree and add points to it
//! let mut kd_tree = KDTree::empty();
//! kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
//! kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
//! kd_tree.add(Point::new_with_coords(&vec![5.0, 6.0]));
//!
//! ```
//!
//! ## KDTree Struct
//!
//! The `KDTree` struct represents the KDTree itself and contains methods for initialization, insertion, and search.
//!
//! - `new(root: KDNode) -> Self`: Creates a new KDTree with the specified root node.
//! - `empty() -> Self`: Creates an empty KDTree.
//! - `add(&mut self, new_point: Point)`: Adds a new point to the KDTree.
//! - `walk(&self, callback: impl Fn(&Point) -> ())`: Performs an in-order traversal of the tree and applies the callback function to each node.
//! - `nearest(&self, target: &Point, n: usize) -> NearestResult`: Finds the nearest neighbors to the target point.
//! - `len(&self) -> usize`: Returns the number of points in the KDTree.
//! - `to_vec(&self) -> PointMatrix`: Converts the KDTree to a vector of points.
//!
//! ## KDNode Struct
//!
//! The `KDNode` struct represents a node in the KDTree and contains methods for node creation, nearest neighbor search, and information retrieval.
//!
//! - `new(point: Point, depth: usize, left: Option<KDNode>, right: Option<KDNode>) -> Self`: Creates a new KDNode with the specified properties.
//! - `from_subtrees(point: Point, depth: usize, left: KDSubTree, right: KDSubTree) -> Self`: Creates a KDNode from left and right subtrees.
//! - `leaf(point: Point, depth: usize) -> Self`: Creates a leaf KDNode with no children.
//! - `nearest(&self, target_point: &Point, best_result: NearestResult) -> NearestResult`: Finds the nearest neighbors to the target point within the subtree rooted at this node.
//! - `left(&self) -> Option<&Box<KDNode>>`: Returns a reference to the left child node.
//! - `right(&self) -> Option<&Box<KDNode>>`: Returns a reference to the right child node.
//! - `is_empty(&self) -> bool`: Returns true if the node is empty.
//! - `is_leaf(&self) -> bool`: Returns true if the node is a leaf.
//! - `len(&self) -> usize`: Returns the number of points in the subtree rooted at this node.
//! - `height(&self) -> usize`: Returns the height of the subtree rooted at this node.
//!
//! ## Usage
//!
//! ```rust
//! use ibn_battuta::algorithms::nearest_neighbor::kd_tree::KDTree;
//! use ibn_battuta::utils::point::Point;
//!
//! // Create a KDTree and add points to it
//! let mut kd_tree = KDTree::empty();
//! kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
//! kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
//! ```


use std::cell::RefCell;
use std::cmp::Ordering;

use getset::{CopyGetters, MutGetters};

use crate::algorithms::nearest_neighbor::nearest_neighbor::{KDSubTree, NearestResult, PointMatrix};
use crate::utils::point::Point;

#[derive(Debug, CopyGetters, MutGetters)]
pub struct KDTree {
    pub(crate) size: usize,
    #[getset(get_copy = "pub", get_mut = "pub")]
    dimensionality: usize,
    pub(crate) root: KDSubTree,
}

impl KDTree {
    /// Creates a new KDTree with the specified root node.
    ///
    /// # Arguments
    ///
    /// * `root` - The root node of the KDTree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDNode, KDTree, Point};
    ///
    /// let root = KDNode::leaf(Point::new_with_coords(&vec![1.0, 2.0]), 0);
    /// let kd_tree = KDTree::new(root);
    /// ```
    pub fn new(root: KDNode) -> Self {
        let pt_dimension = root.point.dimensionality();

        KDTree {
            root: Some(Box::new(root)),
            dimensionality: pt_dimension,
            size: 1,
        }
    }
    /// Creates an empty KDTree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::KDTree;
    ///
    /// let kd_tree = KDTree::empty();
    /// ```
    pub fn empty() -> Self {
        KDTree {
            root: None,
            dimensionality: 0,
            size: 0,
        }
    }

    /// Adds a new point to the KDTree.
    ///
    /// # Arguments
    ///
    /// * `new_point` - The point to be added to the KDTree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// ```
    pub fn add(&mut self, new_point: Point) {
        self.size += 1;
        if self.dimensionality == 0 {
            // Represents the result of a nearest neighbor search.
            self.dimensionality = new_point.dimensionality();
        }

        let parent = std::mem::replace(&mut self.root, None);
        self.root = self.add_rec(parent, new_point, 0);
    }


    /// Recursively adds a new point to the KDTree.
    ///
    /// # Arguments
    ///
    /// * `parent` - The parent node of the current subtree.
    /// * `new_point` - The point to be added to the KDTree.
    /// * `depth` - The current depth of the subtree.
    ///
    /// # Returns
    ///
    /// The updated subtree after adding the new point.
    ///
    /// # Panics
    ///
    /// Panics if the dimensionality of the new point does not match the tree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// ```
    pub fn add_rec(&mut self, parent: KDSubTree, new_point: Point, depth: usize) -> KDSubTree {
        if parent.is_none() {
            return Some(Box::new(KDNode::leaf(new_point, depth + 1)));
        }

        let mut node = parent.unwrap();
        match node.cmp_by_point(&new_point) {
            None => panic!("Point dimensionality is not matching with tree"), // fails with broken data
            Some(Ordering::Greater) => {
                // if parent is greater than new point then the newpoint should go left
                node.left = self.add_rec(node.left, new_point, depth + 1);
                Some(node)
            }
            _ => {
                node.right = self.add_rec(node.right, new_point, depth + 1);
                Some(node)
            }
        }
    }

    /// Performs an in-order traversal of the tree and applies the callback function to each node.
    ///
    /// # Arguments
    ///
    /// * `callback` - The callback function to be applied to each node.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
    ///
    /// kd_tree.walk(|point| {
    ///     println!("Visiting node: {:?}", point);
    /// });
    /// ```
    pub fn walk(&self, callback: impl Fn(&Point) -> ()) {
        self.walk_in_order(&self.root, &callback);
    }

    /// Recursively performs an in-order traversal of the subtree and applies the callback function to each node.
    ///
    /// # Arguments
    ///
    /// * `subtree` - The current subtree to be traversed.
    /// * `callback` - The callback function to be applied to each node.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
    ///
    /// kd_tree.walk_in_order(&kd_tree.root(), &|point| {
    ///     println!("Visiting node: {:?}", point);
    /// });
    /// ```
    pub fn walk_in_order(&self, subtree: &KDSubTree, callback: &impl Fn(&Point) -> ()) {
        if let Some(node) = subtree {
            self.walk_in_order(&node.left, callback);
            callback(&node.point);
            self.walk_in_order(&node.right, callback);
        }
    }

    /// Finds the nearest neighbors to the target point.
    ///
    /// # Arguments
    ///
    /// * `target` - The target point for which nearest neighbors are to be found.
    /// * `n` - The number of nearest neighbors to find.
    ///
    /// # Returns
    ///
    /// The result of the nearest neighbor search, containing information about the closest points.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
    ///
    /// let target_point = Point::new_with_coords(&vec![2.0, 3.0]);
    /// let nearest_result = kd_tree.nearest(&target_point, 1);
    ///
    /// println!("Nearest neighbor: {:?}", nearest_result.closest_point());
    /// ```
    pub fn nearest(&self, target: &Point, n: usize) -> NearestResult {
        let best_result = NearestResult::new(target.clone(), f64::INFINITY, n);

        match &self.root {
            None => best_result,
            Some(n) => n.nearest(target, best_result),
        }
    }

    /// Returns the number of points in the KDTree.
    ///
    /// # Returns
    ///
    /// The number of points in the KDTree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
    ///
    /// let len = kd_tree.len();
    /// println!("Number of points in KDTree: {}", len);
    /// ```
    pub fn len(&self) -> usize {
        self.size
    }

    /// Converts the KDTree to a vector of points.
    ///
    /// # Returns
    ///
    /// A vector containing the points in the KDTree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDTree, Point};
    ///
    /// let mut kd_tree = KDTree::empty();
    /// kd_tree.add(Point::new_with_coords(&vec![1.0, 2.0]));
    /// kd_tree.add(Point::new_with_coords(&vec![3.0, 4.0]));
    ///
    /// let point_matrix = kd_tree.to_vec();
    /// println!("KDTree as a matrix: {:?}", point_matrix);
    /// ```
    pub fn to_vec(&self) -> PointMatrix {
        let pts: RefCell<PointMatrix> = RefCell::new(vec![]);

        self.walk(|n| pts.borrow_mut().push(n.coords().to_vec()));

        let res = pts.borrow().to_vec();
        res
    }
}

/// A node in a KDTree.
#[derive(Debug)]
pub struct KDNode {
    point: Point,
    depth: usize,
    size: usize,
    // todo: remove seems redundant
    left: KDSubTree,
    right: KDSubTree,
}

impl KDNode {
    /// Creates a new KDNode with the specified point, depth, left subtree, and right subtree.
    ///
    /// # Arguments
    ///
    /// * `point` - The point stored in the KDNode.
    /// * `depth` - The depth of the KDNode in the tree.
    /// * `left` - An optional left subtree.
    /// * `right` - An optional right subtree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDNode, Point};
    ///
    /// let left_node = KDNode::leaf(Point::new_with_coords(&vec![1.0, 2.0]), 0);
    /// let right_node = KDNode::leaf(Point::new_with_coords(&vec![3.0, 4.0]), 0);
    ///
    /// let node = KDNode::new(Point::new_with_coords(&vec![2.0, 3.0]), 1, Some(left_node), Some(right_node));
    /// ```
    pub fn new(point: Point, depth: usize, left: Option<KDNode>, right: Option<KDNode>) -> Self {
        let left_node = match left {
            Some(node) => Some(Box::new(node)),
            None => None,
        };

        let right_node = match right {
            Some(node) => Some(Box::new(node)),
            None => None,
        };

        let left_size = left_node.as_ref().map_or(0, |n| n.len());
        let right_size = right_node.as_ref().map_or(0, |n| n.len());

        KDNode {
            point,
            depth,
            size: 1 + left_size + right_size,
            left: left_node,
            right: right_node,
        }
    }

    /// Creates a new KDNode from existing subtrees.
    ///
    /// # Arguments
    ///
    /// * `point` - The point stored in the KDNode.
    /// * `depth` - The depth of the KDNode in the tree.
    /// * `left` - The left subtree.
    /// * `right` - The right subtree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDNode, Point};
    ///
    /// let left_subtree = Some(Box::new(KDNode::leaf(Point::new_with_coords(&vec![1.0, 2.0]), 0)));
    /// let right_subtree = Some(Box::new(KDNode::leaf(Point::new_with_coords(&vec![3.0, 4.0]), 0)));
    ///
    /// let node = KDNode::from_subtrees(Point::new_with_coords(&vec![2.0, 3.0]), 1, left_subtree, right_subtree);
    /// ```
    pub fn from_subtrees(point: Point, depth: usize, left: KDSubTree, right: KDSubTree) -> Self {
        let left_size = left.as_ref().map_or(0, |n| n.len());
        let right_size = right.as_ref().map_or(0, |n| n.len());

        KDNode {
            point,
            depth,
            left,
            right,
            size: 1 + left_size + right_size,
        }
    }

    /// Creates a new leaf KDNode with the specified point and depth.
    ///
    /// # Arguments
    ///
    /// * `point` - The point stored in the KDNode.
    /// * `depth` - The depth of the KDNode in the tree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDNode, Point};
    ///
    /// let leaf_node = KDNode::leaf(Point::new_with_coords(&vec![2.0, 3.0]), 1);
    /// ```
    pub fn leaf(point: Point, depth: usize) -> Self {
        KDNode {
            point,
            depth,
            size: 1,
            left: None,
            right: None,
        }
    }

    /// Finds the nearest neighbors to the target point.
    ///
    /// # Arguments
    ///
    /// * `target_point` - The target point for which nearest neighbors are to be found.
    /// * `best_result` - The current best result of the nearest neighbor search.
    ///
    /// # Returns
    ///
    /// The updated result of the nearest neighbor search.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ibn_battuta::{KDNode, Point, NearestResult};
    ///
    /// let leaf_node = KDNode::leaf(Point::new_with_coords(&vec![2.0, 3.0]), 1);
    /// let target_point = Point::new_with_coords(&vec![3.0, 4.0]);
    /// let best_result = NearestResult::new(target_point.clone(), f64::INFINITY, 1);
    ///
    /// let nearest_result = leaf_node.nearest(&target_point, best_result);
    /// ```
    pub fn nearest(&self, target_point: &Point, best_result: NearestResult) -> NearestResult {
        if self.is_empty() {
            return best_result;
        }

        let distance_from_target = self.point.distance(target_point);

        let best_distance = best_result.distance;
        let mut nearest_result = best_result;

        if distance_from_target <= best_distance {
            let pt = self.point.clone();
            nearest_result.add(pt, distance_from_target);
        };

        let (closest_branch, futher_branch) = match self.cmp_by_point(&target_point) {
            None => panic!("Dimension conflict in nearest function"),
            Some(Ordering::Greater) => (self.left(), self.right()),
            Some(_) => (self.right(), self.left()),
        };

        if closest_branch.is_some() {
            let closest_result = closest_branch
                .unwrap()
                .nearest(target_point, nearest_result.clone());

            let pt_distance = closest_result.closest_distance();
            nearest_result.add(closest_result.point, pt_distance);
        }

        // check distance from split line
        let split_dist = self.point.split_distance(&target_point, self.level_coord());
        if nearest_result.closest_distance() > split_dist && futher_branch.is_some() {
            let further_result = futher_branch
                .unwrap()
                .nearest(target_point, nearest_result.clone());
            let pt_distance = further_result.closest_distance();
            nearest_result.add(further_result.point, pt_distance);
        }

        nearest_result
    }

    // fn point(&self) -> &Point {
    //     &self.point
    // }

    // fn cmp(&self, other: &KDNode) -> Option<Ordering> {
    //     self.cmp_by_point(&other.point)
    // }

    /// Compares the current KDNode's point to another point based on the current dimension.
    ///
    /// This method is used to determine the ordering of points along a specific dimension
    /// during nearest neighbor searches.
    ///
    /// # Arguments
    ///
    /// * `other` - The point to compare against.
    ///
    /// # Returns
    ///
    /// An `Option<Ordering>` representing the result of the comparison.
    /// - `Some(Ordering::Less)` if the current point is less than the other point along the current dimension.
    /// - `Some(Ordering::Equal)` if the points are equal along the current dimension.
    /// - `Some(Ordering::Greater)` if the current point is greater than the other point along the current dimension.
    /// - `None` if there is a dimension conflict between the points.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::cmp::Ordering;
    /// use ibn_battuta::{KDNode, Point};
    ///
    /// let node = KDNode::leaf(Point::new_with_coords(&vec![1.0, 2.0]), 0);
    /// let other_point = Point::new_with_coords(&vec![3.0, 4.0]);
    ///
    /// match node.cmp_by_point(&other_point) {
    ///     Some(Ordering::Less) => println!("Current point is less along the current dimension."),
    ///     Some(Ordering::Equal) => println!("Points are equal along the current dimension."),
    ///     Some(Ordering::Greater) => println!("Current point is greater along the current dimension."),
    ///     None => println!("Dimension conflict between points."),
    /// }
    /// ```
    pub fn cmp_by_point(&self, other: &Point) -> Option<Ordering> {
        self.point.cmp_by_coord(other, self.level_coord())
    }

    /// returns a dimension for comparision
    fn level_coord(&self) -> usize {
        self.depth % self.point.dimensionality()
    }

    /// Returns a reference to the left subtree.
    pub fn left(&self) -> Option<&Box<KDNode>> {
        self.left.as_ref()
    }


    /// Returns a reference to the right subtree.
    pub fn right(&self) -> Option<&Box<KDNode>> {
        self.right.as_ref()
    }

    /// Checks if the KDNode is empty (contains no points).
    pub fn is_empty(&self) -> bool {
        if self.len() == 0 {
            true
        } else {
            false
        }
    }

    /// Checks if the KDNode is a leaf node (contains a single point).
    pub fn is_leaf(&self) -> bool {
        self.len() == 1
    }

    /// Returns the number of points in the subtree rooted at this node.
    ///
    /// # Returns
    ///
    /// The number of points in the subtree rooted at this node.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns the number of levels in the subtree rooted at this node;
    /// leaves have height 1.
    ///
    /// # Returns
    ///
    /// The height of the subtree rooted at this node.
    pub fn height(&self) -> usize {
        if self.is_empty() {
            0
        } else if self.is_leaf() {
            1
        } else {
            let left_height = self.left().map_or(0, |n| n.height());
            let right_height = self.right().map_or(0, |n| n.height());

            1 + std::cmp::max(left_height, right_height)
        }
    }
}