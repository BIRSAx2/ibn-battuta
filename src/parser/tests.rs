#![cfg(test)]
use crate::{metric::*, Tsp, WeightFormat};
use crate::{TspBuilder, TspKind, WeightKind};
use approx::assert_relative_eq;

const TEST_STR: &str = "
NAME: test
TYPE: TSP
COMMENT: Test
DIMENSION: 3
EDGE_WEIGHT_TYPE: GEO
DISPLAY_DATA_TYPE: COORD_DISPLAY
NODE_COORD_SECTION
1 38.24 20.42
2 39.57 26.15
3 40.56 25.32
EOF
";

#[test]
fn test_read_str() {
    let result = TspBuilder::parse_str(TEST_STR);
    assert!(result.is_ok());
    let tsp = result.unwrap();

    assert_eq!(3, tsp.dim());
    assert_eq!(TspKind::Tsp, tsp.kind());
    assert_eq!(WeightKind::Geo, tsp.weight_kind());
}

#[test]
fn test_read_str_missing_name() {
    let mut s = String::from("");
    for (idx, line) in TEST_STR.lines().enumerate() {
        if idx == 1 {
            continue;
        }
        s.push_str(line);
        s.push('\n');
    }

    let result = TspBuilder::parse_str(s);
    assert!(result.is_err());
}

#[test]
fn test_read_str_missing_type() {
    let mut s = String::from("");
    for (idx, line) in TEST_STR.lines().enumerate() {
        if idx == 2 {
            continue;
        }
        s.push_str(line);
        s.push('\n');
    }

    let result = TspBuilder::parse_str(s);
    assert!(result.is_err());
}

#[test]
fn test_read_str_missing_dim() {
    let mut s = String::from("");
    for (idx, line) in TEST_STR.lines().enumerate() {
        if idx == 4 {
            continue;
        }
        s.push_str(line);
        s.push('\n');
    }

    let result = TspBuilder::parse_str(s);
    assert!(result.is_err());
}

#[test]
fn test_read_str_missing_wtype() {
    let mut s = String::from("");
    for (idx, line) in TEST_STR.lines().enumerate() {
        if idx == 5 {
            continue;
        }
        s.push_str(line);
        s.push('\n');
    }

    let result = TspBuilder::parse_str(s);
    assert!(result.is_err());
}

#[allow(unused_macros)]
macro_rules! prep_weight {
    ($x:expr, $w:expr) => {
        format!(
            "
        NAME: test
        TYPE: TSP
        COMMENT: Test
        DIMENSION: 5
        EDGE_WEIGHT_TYPE: EXPLICIT
        EDGE_WEIGHT_FORMAT: {}
        EDGE_WEIGHT_SECTION
        {}
        ",
            $x, $w
        )
    };
}

fn test_weight(tsp: Tsp) {
    assert_relative_eq!(5_f64, tsp.weight(1, 2));
    assert_relative_eq!(10., tsp.weight(4, 3));
    assert_relative_eq!(10., tsp.weight(3, 4));
    assert_relative_eq!(9., tsp.weight(4, 2));
    assert_relative_eq!(9., tsp.weight(2, 4));
    assert_relative_eq!(0., tsp.weight(4, 4));
}

#[test]
fn test_weight_upper() {
    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::UpperRow.tsp_str(),
        "1 2 3 4 5 6 7 8 9 10"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());

    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::UpperDiagRow.tsp_str(),
        "0 1 2 3 4 0 5 6 7 0 8 9 0 10 0"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());

    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::UpperCol.tsp_str(),
        "1 2 5 3 6 8 4 7 9 10"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());

    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::UpperDiagCol.tsp_str(),
        "0 1 0 2 5 0 3 6 8 0 4 7 9 10 0"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());
}

#[test]
fn test_weight_lower() {
    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::LowerRow.tsp_str(),
        "1 2 5 3 6 8 4 7 9 10"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());

    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::LowerDiagRow.tsp_str(),
        "0 1 0 2 5 0 3 6 8 0 4 7 9 10 0"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());

    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::LowerCol.tsp_str(),
        "1 2 3 4 5 6 7 8 9 10"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());

    let result = TspBuilder::parse_str(prep_weight!(
        WeightFormat::LowerDiagCol.tsp_str(),
        "0 1 2 3 4 0 5 6 7 0 8 9 0 10 0"
    ));
    assert!(result.is_ok(), "{}", result.err().unwrap());
    test_weight(result.unwrap());
}

#[test]
fn test_tour() {
    let s1 = "
    NAME : test
    COMMENT : test
    TYPE : TOUR
    DIMENSION : 4
    TOUR_SECTION
    4 3 2 1
    -1
    EOF
    ";

    let result = TspBuilder::parse_str(s1);
    assert!(result.is_ok());
    let tsp = result.unwrap();
    assert_eq!(1, tsp.tours().len());

    let s2 = "
    NAME : test
    COMMENT : test
    TYPE : TOUR
    DIMENSION : 4
    TOUR_SECTION
    1
    2
    3
    4
    -1
    4 3 2 1
    -1
    -1
    EOF
    ";

    let result = TspBuilder::parse_str(s2);
    assert!(result.is_ok());
    let tsp = result.unwrap();
    assert_eq!(2, tsp.tours().len());
}

#[test]
fn test_metric_fn() {
    assert_eq!(5., euc_2d(&vec![6., 0.], &vec![3., 4.]), "Test euc_2d");
    assert_eq!(
        5. * (2 as f64).sqrt(),
        euc_3d(&vec![6., 0., -2.], &vec![3., 4., 3.]),
        "Test euc_3d"
    );
    assert_eq!(7., man_2d(&vec![6., 0.], &vec![3., 4.]), "Test man_2d");
    assert_eq!(
        12.,
        man_3d(&vec![6., 0., -2.], &vec![3., 4., 3.]),
        "Test man_3d"
    );
    assert_eq!(4., max_2d(&vec![6., 0.], &vec![3., 4.]), "Test max_2d");
    assert_eq!(
        5.,
        max_3d(&vec![6., 0., -2.], &vec![3., 4., 3.]),
        "Test max_3d"
    );

    let eps = geo(&vec![89.6, -74.6], &vec![-29.6, -14.6]) - 13359.864588;
    assert!(eps.abs() < 1e-6, "Test geo");
    // 13359.864588
    assert_eq!(
        18000.,
        xray1(&vec![360., 75., -55.], &vec![180., -45., 22.]),
        "Test xray1"
    );
    assert_eq!(
        14400.,
        xray2(&vec![360., 75., -55.], &vec![180., -45., 22.]),
        "Test xray2"
    );
}