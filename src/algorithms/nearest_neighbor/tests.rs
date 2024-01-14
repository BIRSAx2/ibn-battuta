#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::cmp::Ordering;

    use approx::assert_relative_eq;

    use crate::algorithms::nearest_neighbor::kd_tree::{KDNode, KDTree};
    use crate::algorithms::nearest_neighbor::nearest_neighbor::{build_points, from_cities, partition_points, PointMatrix};
    use crate::utils::point::Point;

    #[test]
    fn point_cmp_by_coord_with_empty_points() {
        let first_pt = Point::new_with_coords(&[]);
        let other_pt = Point::new_with_coords(&[]);

        assert_eq!(None, first_pt.cmp_by_coord(&other_pt, 0));
    }

    #[test]
    fn point_cmp_by_coord_when_other_node_has_different_dim() {
        let first_pt = Point::new_with_coords(&[1.0, 0.0]);
        let other_pt = Point::new_with_coords(&[0.0]);

        assert_eq!(None, first_pt.cmp_by_coord(&other_pt, 1));
    }

    #[test]
    fn point_cmp_by_coord_with_pt_less_than() {
        let pt = Point::new_with_coords(&[-1.0, 0.0]);
        let other_pt = Point::new_with_coords(&[0.0, -1.0]);

        assert_eq!(Some(Ordering::Less), pt.cmp_by_coord(&other_pt, 0));
        assert_eq!(Some(Ordering::Less), other_pt.cmp_by_coord(&pt, 1));
    }

    #[test]
    fn point_cmp_by_coord_with_pt_equal() {
        let pt = Point::new_with_coords(&[-1.0, 0.0]);

        assert_eq!(Some(Ordering::Equal), pt.cmp_by_coord(&pt, 0));
    }

    #[test]
    fn point_cmp_by_coord_with_pt_greater_than() {
        let pt = Point::new_with_coords(&[1.0, 0.0]);
        let other_pt = Point::new_with_coords(&[0.0, 1.0]);

        assert_eq!(Some(Ordering::Greater), pt.cmp_by_coord(&other_pt, 0));
        assert_eq!(Some(Ordering::Greater), other_pt.cmp_by_coord(&pt, 1));
    }

    #[test]
    fn point_eq_with_same_values() {
        let pt = Point::new_with_coords(&[1.0, 0.0]);

        assert!(pt.eq(&pt));
    }

    #[test]
    fn point_distance_from_origin_to_origin() {
        let pt = Point::new_with_coords(&[0.0, 0.0]);

        assert_relative_eq!(0.0, pt.distance(&pt));
    }

    #[test]
    fn point_distance_from_origin_to_x_axis() {
        let origin = Point::new_with_coords(&[0.0, 0.0]);
        let other = Point::new_with_coords(&[1.0, 0.0]);

        assert_relative_eq!(1.0, origin.distance(&other));
    }

    #[test]
    fn point_distance_from_origin_to_y_axis() {
        let origin = Point::new_with_coords(&[0.0, 0.0]);
        let other = Point::new_with_coords(&[0.0, 1.0]);

        assert_relative_eq!(1.0, origin.distance(&other));
    }

    #[test]
    fn point_distance_on_diagonal() {
        let pt = Point::new_with_coords(&[-1.0, -1.0]);
        let other = Point::new_with_coords(&[1.0, 1.0]);

        assert_relative_eq!(2.828427, pt.distance(&other))
    }

    #[test]
    fn kdtree_add_new_node_to_empty_tree() {
        let mut tree = KDTree::empty();

        assert_eq!(0, tree.len());

        tree.add(Point::new_with_coords(&[0.0, 0.0]));

        assert_eq!(1, tree.len());
        assert_eq!(2, tree.dimensionality());
    }

    #[test]
    fn kdtree_add_new_node_to_tree_with_root() {
        let root = KDNode::new(Point::new_with_coords(&[0.0, 0.0]), 0, None, None);

        let mut tree = KDTree::new(root);

        assert_eq!(1, tree.len());
        assert_eq!(2, tree.dimensionality());

        tree.add(Point::new_with_coords(&[-1.0, 0.0]));

        assert_eq!(2, tree.len());
        assert_eq!(2, tree.dimensionality());
    }

    #[test]
    fn kdtree_walk_with_empty_tree() {
        let tree = KDTree::empty();

        let pts: RefCell<Vec<Point>> = RefCell::new(vec![]);
        tree.walk(|pt| pts.borrow_mut().push(pt.clone()));

        assert!(pts.borrow().is_empty());
    }

    #[test]
    fn kdtree_walk_with_only_root_node() {
        let root = KDNode::new(Point::new_with_coords(&[0.0, 0.0]), 0, None, None);
        let tree = KDTree::new(root);

        let pts: RefCell<Vec<Point>> = RefCell::new(vec![]);
        tree.walk(|pt| pts.borrow_mut().push(pt.clone()));

        assert!(!pts.borrow().is_empty());
        assert_eq!(1, pts.borrow().len());
        assert_eq!(&[0.0, 0.0], pts.borrow().get(0).unwrap().coords());
    }

    #[test]
    fn kdtree_walk_balances_1level_tree() {
        let mut tree = KDTree::empty();

        //add some nodes
        tree.add(Point::new_with_coords(&[0.0, 0.0]));
        tree.add(Point::new_with_coords(&[-1.0, 0.0]));
        tree.add(Point::new_with_coords(&[1.0, 0.0]));

        // double-check insertion
        assert_eq!(3, tree.len());
        assert_eq!(2, tree.dimensionality());

        // check insertion order
        let pts: RefCell<Vec<Point>> = RefCell::new(vec![]);
        tree.walk(|pt| pts.borrow_mut().push(pt.clone()));

        assert_eq!(3, pts.borrow().len());
        assert_eq!(&[0.0, 0.0], pts.borrow().get(0).unwrap().coords());
        assert_eq!(&[-1.0, 0.0], pts.borrow().get(1).unwrap().coords());
        assert_eq!(&[1.0, 0.0], pts.borrow().get(2).unwrap().coords());
    }

    #[test]
    fn partition_points_single_elem() {
        let points = build_points(&[vec![0.0, 0.0]]);

        let res = partition_points(points, 0, 2);
        assert_eq!(&[0.0, 0.0], res.0.coords());
        assert!(res.1.is_empty());
        assert!(res.2.is_empty());
    }

    #[test]
    fn partition_points_with_2points_with_left_subtree() {
        let points = build_points(&[vec![-1.0, 0.0], vec![0.0, 0.0]]);

        let res = partition_points(points, 0, 2);
        assert_eq!(&[0.0, 0.0], res.0.coords());
        assert_eq!(&[-1.0, 0.0], res.1[0].coords());
        assert!(res.2.is_empty());
    }

    #[test]
    fn partition_points_with_2points_with_right_subtree() {
        let points = build_points(&vec![vec![0.0, 0.0], vec![2.0, 0.0]]);

        let res = partition_points(points, 0, 2);
        assert_eq!(&[2.0, 0.0], res.0.coords());
        assert_eq!(&[0.0, 0.0], res.1[0].coords());
        assert!(res.2.is_empty());
    }

    #[test]
    fn partition_points_with_2points_with_full_tree() {
        let points = build_points(&vec![vec![-1.0, 0.0], vec![2.0, 0.0], vec![0.0, 0.0]]);

        let res = partition_points(points, 0, 2);
        assert_eq!(&[0.0, 0.0], res.0.coords());
        assert_eq!(&[-1.0, 0.0], res.1[0].coords());
        assert_eq!(&[2.0, 0.0], res.2[0].coords());
    }

    #[test]
    fn partition_points_with_3points_by_second_dimension() {
        let points = build_points(&vec![vec![0.0, 0.0], vec![2.0, -1.0], vec![1.0, 2.0]]);

        let res = partition_points(points, 1, 2);
        assert_eq!(&[0.0, 0.0], res.0.coords());
        assert_eq!(&[2.0, -1.0], res.1[0].coords());
        assert_eq!(&[1.0, 2.0], res.2[0].coords());
    }

    #[test]
    fn from_cities_example() {
        let points = build_points(&vec![
            vec![0.0, 0.0],
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            vec![-1.0, -1.0],
            vec![-1.0, 1.0],
            vec![1.0, -1.0],
            vec![1.0, 1.0],
        ]);
        let tree = from_cities(&points);

        assert_eq!(7, tree.len());

        let points: RefCell<PointMatrix> = RefCell::new(vec![]);
        tree.walk(|n| points.borrow_mut().push(n.coords().to_vec()));

        assert_eq!(vec![-1.0, -1.0], points.borrow()[0]);
        assert_eq!(vec![-1.0, 0.0], points.borrow()[1]);
        assert_eq!(vec![-1.0, 1.0], points.borrow()[2]);
        assert_eq!(vec![0.0, 0.0], points.borrow()[3]);
        assert_eq!(vec![1.0, -1.0], points.borrow()[4]);
        assert_eq!(vec![1.0, 0.0], points.borrow()[5]);
        assert_eq!(vec![1.0, 1.0], points.borrow()[6]);
    }

    #[test]
    fn kdtree_nearest_for_tsp_5_1() {
        let cities = build_points(&[
            vec![0.0, 0.0],
            vec![0.0, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
        ]);

        let kd = from_cities(&cities);

        let res = kd.nearest(&cities[0], 2);
        assert_eq!(cities[1].id(), res.point.id());

        let res2 = kd.nearest(&cities[1], 2);
        assert_eq!(cities[2].id(), res2.point.id());

        let res3 = kd.nearest(&cities[2], 2);
        assert_eq!(cities[1].id(), res3.point.id());

        let res4 = kd.nearest(&cities[3], 2);
        assert_eq!(cities[2].id(), res4.point.id());

        let res5 = kd.nearest(&cities[4], 2);
        assert_eq!(cities[3].id(), res5.point.id());
    }

    #[test]
    fn kdtree_nearest_with_points_around_node4() {
        let points = build_points(&vec![
            vec![100.0, 100.0],
            vec![-100.0, 100.0],
            vec![100.0, -100.0],
            vec![-100.0, -100.0], // it is node 4
        ]);

        let expected_coords = vec![-100.0, -100.0];
        let tree = from_cities(&points);
        assert_eq!(4, tree.len());

        let pt1 = Point::new_with_coords(&[-110.0, -100.0]);
        let res = tree.nearest(&pt1, 1);

        assert_relative_eq!(10.0, res.distance);
        assert_eq!(expected_coords, res.point.coords());

        let pt2 = Point::new_with_coords(&[-90.0, -100.0]);
        let res = tree.nearest(&pt2, 1);

        assert_relative_eq!(10.0, res.distance);
        assert_eq!(expected_coords, res.point.coords());

        let pt3 = Point::new_with_coords(&[-100.0, -90.0]);
        let res = tree.nearest(&pt3, 1);

        assert_relative_eq!(10.0, res.distance);
        assert_eq!(expected_coords, res.point.coords());

        let pt4 = Point::new_with_coords(&[-100.0, -110.0]);
        let res = tree.nearest(&pt4, 1);

        assert_relative_eq!(10.0, res.distance);
        assert_eq!(expected_coords, res.point.coords());
    }
}