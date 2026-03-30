use std::f64::consts::PI;

use crate::WeightKind;

const EARTH_RADIUS: f64 = 6378.388;

impl WeightKind {
    /// Calculates and returns the cost (or distance) between two points.
    ///
    /// For [`WeightKind::Custom`] and [`WeightKind::Undefined`], the function will always return ```0.```.
    pub fn cost(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Self::Euc2d => euc_2d(a, b),
            Self::Euc3d => euc_3d(a, b),
            Self::Geo => geo(a, b),
            Self::Max2d => max_2d(a, b),
            Self::Max3d => max_3d(a, b),
            Self::Man2d => man_2d(a, b),
            Self::Man3d => man_3d(a, b),
            Self::Ceil2d => euc_2d(a, b).round(),
            Self::Att => att(a, b),
            Self::Xray1 => xray1(a, b),
            Self::Xray2 => xray2(a, b),
            _ => 0.,
        }
    }
}

/// Calculates the 2D-Euclidean distance between two points.
#[inline]
pub fn euc_2d(a: &[f64], b: &[f64]) -> f64 {
    euc(a, b, 2)
}

/// Calculates the 3D-Euclidean distance between two points.
#[inline]
pub fn euc_3d(a: &[f64], b: &[f64]) -> f64 {
    euc(a, b, 3)
}

#[inline]
fn euc(a: &[f64], b: &[f64], k: usize) -> f64 {
    a.iter()
        .take(k)
        .zip(b.iter().take(k))
        .fold(0_f64, |acc, (x1, x2)| acc + (x1 - x2).powi(2))
        .sqrt()
}

/// Calculates the 2D-Manhattan distance between two points.
#[inline]
pub fn man_2d(a: &[f64], b: &[f64]) -> f64 {
    man(a, b, 2)
}

/// Calculates the 3D-Manhattan distance between two points.
#[inline]
pub fn man_3d(a: &[f64], b: &[f64]) -> f64 {
    man(a, b, 3)
}

#[inline]
fn man(a: &[f64], b: &[f64], k: usize) -> f64 {
    a.iter()
        .take(k)
        .zip(b.iter().take(k))
        .fold(0_f64, |acc, (x1, x2)| acc + (x1 - x2).abs())
}

/// Calculates the 2D maximum distance between two points.
#[inline]
pub fn max_2d(a: &[f64], b: &[f64]) -> f64 {
    max(a, b, 2)
}

/// Calculates the 3D maximum distance between two points.
#[inline]
pub fn max_3d(a: &[f64], b: &[f64]) -> f64 {
    max(a, b, 3)
}

#[inline]
fn max(a: &[f64], b: &[f64], k: usize) -> f64 {
    a.iter()
        .take(k)
        .zip(b.iter().take(k))
        .fold(0_f64, |acc, (x1, x2)| acc.max((x1 - x2).abs()))
}

/// Calculates the geographical between two points.
#[inline]
pub fn geo(a: &[f64], b: &[f64]) -> f64 {
    let (lat_a, lon_a) = (to_geo_coord(a[0]), to_geo_coord(a[1]));
    let (lat_b, lon_b) = (to_geo_coord(b[0]), to_geo_coord(b[1]));

    let q1 = (lon_a - lon_b).cos();
    let q2 = (lat_a - lat_b).cos();
    let q3 = (lat_a + lat_b).cos();
    let q4 = (0.5 * ((1. + q1) * q2 - (1. - q1) * q3)).acos();
    EARTH_RADIUS * q4 + 1.
}

#[inline]
fn to_geo_coord(x: f64) -> f64 {
    let deg = x.trunc();
    let min = x - deg;
    PI * (deg + 5. * min / 3.) / 180.
}

/// Calculates the distance between two points for dataset from AT&T Bell laboratory, published by Padberg and Rinaldi in 1987.
#[inline]
pub fn att(a: &[f64], b: &[f64]) -> f64 {
    (a.iter()
        .take(2)
        .zip(b.iter().take(2))
        .fold(0_f64, |acc, (x1, x2)| acc + (x1 - x2).powi(2))
        / 10.)
        .sqrt()
}

/// Calculates the distance between two points for crystallography problems (version 1).
#[inline]
pub fn xray1(a: &[f64], b: &[f64]) -> f64 {
    let dx = (a[0] - b[0]).abs();
    let pr = dx.min((dx - 360.).abs());
    let dy = (a[1] - b[1]).abs();
    let dz = (a[2] - b[2]).abs();
    100. * pr.max(dy.max(dz))
}

/// Calculates the distance between two points for crystallography problems (version 2).
#[inline]
pub fn xray2(a: &[f64], b: &[f64]) -> f64 {
    let dx = (a[0] - b[0]).abs();
    let pr = dx.min((dx - 360.).abs());
    let dy = (a[1] - b[1]).abs();
    let dz = (a[2] - b[2]).abs();
    100. * (pr / 1.25).max((dy / 1.5).max(dz / 1.15))
}
