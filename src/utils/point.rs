use std::cmp::Ordering;

/// Represents a node coordinate.
#[derive(Clone, Debug)]
pub struct Point {
    /// Id of a point.
    id: usize,
    /// Point's coordinates.
    coordinates: Vec<f64>,
}


impl Point {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn coordinates(&self) -> &Vec<f64> {
        &self.coordinates
    }

    pub fn into_value(self) -> (usize, Vec<f64>) {
        (self.id, self.coordinates)
    }

    /// Constructs a new point.
    pub fn new(id: usize, pos: Vec<f64>) -> Self {
        Self { id, coordinates: pos }
    }

    pub fn new2(id: usize, x: f64, y: f64) -> Self {
        Self::new(id, vec![x, y])
    }

    pub fn new3(id: usize, x: f64, y: f64, z: f64) -> Self {
        Self::new(id, vec![x, y, z])
    }


    pub fn new_with_coords(coords: &[f64]) -> Self {
        Point::new(0, coords.to_vec())
    }
    pub fn new_with_id(id: usize, coords: &[f64]) -> Self {
        Point::new(id, coords.to_vec())
    }

    pub fn dimensionality(&self) -> usize {
        self.coordinates().len()
    }

    pub fn coords(&self) -> &[f64] {
        &self.coordinates()[..]
    }

    pub fn get(&self, dimension: usize) -> Option<f64> {
        self.coordinates().get(dimension).map(|x| x.clone())
    }

    // TODO: finish and add tests
    pub fn eq(&self, other: &Point) -> bool {
        if self.dimensionality() != other.dimensionality() {
            return false;
        }

        let diff = self.distance(other);
        diff < f64::EPSILON
    }

    pub fn distance(&self, other: &Point) -> f64 {
        let distance: f64 = self
            .coordinates()
            .iter()
            .zip(other.coords())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();

        distance
    }

    /// returns distance from split level
    pub fn split_distance(&self, other: &Point, coord: usize) -> f64 {
        (self.coordinates()[coord] - other.get(coord).unwrap()).abs()
    }

    pub fn cmp_by_coord(&self, other: &Point, coord: usize) -> Option<Ordering> {
        if self.get(coord).is_none() || other.get(coord).is_none() {
            return None;
        }

        let self_coord = self.get(coord).unwrap();
        let other_coord = other.get(coord).unwrap();

        let res = if self_coord < other_coord {
            Ordering::Less
        } else if (self_coord - other_coord).abs() < f64::EPSILON {
            Ordering::Equal
        } else {
            Ordering::Greater
        };

        Some(res)
    }

    pub fn x(&self) -> f64 {
        if self.dimensionality() < 1 {
            panic!("for accessing y, dimensionality must be > 0");
        }

        self.coordinates()[0].clone()
    }

    pub fn y(&self) -> f64 {
        if self.dimensionality() < 2 {
            panic!("for accessing y, dimensionality must be > 1");
        }

        self.coordinates()[1].clone()
    }
}