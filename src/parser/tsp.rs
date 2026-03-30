use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use getset::{CopyGetters, Getters, MutGetters};

use crate::parser::ParseTspError;

// (Some) keywords for data specification part.
static K_NAME: &str = "NAME";
static K_TYPE: &str = "TYPE";
static K_DIM: &str = "DIMENSION";
static K_CAP: &str = "CAPACITY";
static K_WEIGHT_TYPE: &str = "EDGE_WEIGHT_TYPE";
static K_WEIGHT_FORMAT: &str = "EDGE_WEIGHT_FORMAT";
static K_EDGE_FORMAT: &str = "EDGE_DATA_FORMAT";
static K_NODE_COORD_TYPE: &str = "NODE_COORD_TYPE";
static K_DISP_TYPE: &str = "DISPLAY_DATA_TYPE";

// (Some) keywords for the data part.
static K_NODE_COORD_SEC: &str = "NODE_COORD_SECTION";
static K_EDGE_WEIGHT_SEC: &str = "EDGE_WEIGHT_SECTION";
static K_TOUR_SEC: &str = "TOUR_SECTION";

/// Represents a parsed TSP dataset.
///
/// An instance of this struct can only be created through the use of [`TspBuilder`].
///
/// TSP format has two parts:
/// - *Specification part*: contains metadata and general information about the dataset.
/// - *Data part*: contains all data stored according to the formats given in the specification part.
///
/// # Format
/// The *specification part* has the following entries:
/// - ```NAME``` (required): the name of a dataset.
/// - ```TYPE``` (required): the type specifier of a dataset. Represented by the enum [`TspKind`].
/// - ```COMMENT``` (optional): comments of a dataset.
/// - ```DIM``` (required): the dimension of a dataset.
/// - ```CAPACITY``` (required if ```TYPE``` is [`TspKind::Cvrp`]): the truck capacity in Capacitated
///   Vehicle Routing Problem (CVRP).
/// - ```EDGE_WEIGHT_TYPE``` (required): indicates how the edge weights (or distances) are calculated.
///   Represented by the enum [`WeightKind`].
/// - ```EDGE_WEIGHT_FORMAT``` (required if ```EDGE_WEIGHT_TYPE``` is [`WeightKind::Explicit`]):
///   specifies how the edge weights are given in the file. Represented by the enum [`WeightFormat`].
/// - ```EDGE_DATA_FORMAT``` (optional): specifies how the edges of a graph are given in the file,
///   if the graph is not complete. Represented by the enum [`EdgeFormat`].
/// - ```NODE_COORD_TYPE``` (required if ```EDGE_WEIGHT_TYPE``` is not [`WeightKind::Explicit`]):
///   specifies how the coordinate for each node is given in the file. Represented by the enum [`CoordKind`].
/// - ```DISPLAY_DATA_TYPE``` (optional): spcifies how the coordinate for each node for display
///   purpose is given in the file. Represented by the enum [`DisplayKind`].
///
/// The *data part* has the following entries:
/// - ```NODE_COORD_SECTION``` (required if ```NODE_COORD_TYPE``` is not [`CoordKind::NoCoord`]):
///   a list of node coordinates.
/// - ```DEPOT_SECTION``` (relevant for [`TspKind::Cvrp`]): a list of possible alternate nodes.
/// - ```DEMAND_SECTION``` (relevant for [`TspKind::Cvrp`]): a list of demands for all nodes. Each
///   entry is a tuple ```(usize, usize)```, in which the first number is a node's id and the second
///   number represents the demand for that node. All depot nodes must be also included in this section
///   and their demands are always ```0```.
/// - ```EDGE_DATA_SECTION```: a list of edges.
/// - ```FIXED_EDGES_SECTION``` (optional): a list of edges that must be included in solutions to the problem.
/// - ```DISPLAY_DATA_SECTION``` (required if ```DISPLAY_DATA_TYPE``` is [`DisplayKind::Disp2d`]):
///   a list of 2D node coordinates for display purpose.
/// - ```TOUR_SECTION```: a collection of tours. Each tour is a sequence of node ids.
/// - ```EDGE_WEIGHT_SECTION```(optional if ```EDGE_WEIGHT_FORMAT``` is [`WeightFormat::Function`]):
///   node coordinates in a matrix form as dictated in ```EDGE_WEIGHT_FORMAT```.
///
/// # Example
///
/// The following example shows how to parse a TSP data from string with [`TspBuilder::parse_str`]:
///
/// ```
/// use ibn_battuta::TspBuilder;
///
/// let s = "
/// NAME: test
/// TYPE: TSP
/// COMMENT: Test
/// DIMENSION: 3
/// EDGE_WEIGHT_TYPE: GEO
/// DISPLAY_DATA_TYPE: COORD_DISPLAY
/// NODE_COORD_SECTION
/// 1 38.24 20.42
/// 2 39.57 26.15
/// 3 40.56 25.32
/// EOF
/// ";
/// let result = TspBuilder::parse_str(s);
/// assert!(result.is_ok());
/// let _ = result.unwrap();
/// ```
///
/// We can also parse a file by calling the function [`TspBuilder::parse_path`]:
///
/// ```no_run
/// use ibn_battuta::parser::TspBuilder;
/// use std::path::Path;
///
/// let path = Path::new("./data/tsplib/berlin52.tsp");
/// let result = TspBuilder::parse_path(path);
/// assert!(result.is_ok());
/// ```
#[derive(Debug, CopyGetters, Getters, MutGetters, Clone)]
pub struct Tsp {
    /// Name of the dataset.
    ///
    /// Maps to the entry ```NAME``` in the TSP format.
    #[getset(get = "pub")]
    name: String,
    /// Type specifier of the dataset.
    ///
    /// Maps to the entry ```TYPE``` in the TSP format.
    #[getset(get_copy = "pub")]
    kind: TspKind,
    /// Additional comments.
    ///
    /// Maps to the entry ```COMMENT``` in the TSP format.
    #[getset(get = "pub")]
    comment: String,
    /// The dimension of a dataset.
    ///
    /// Maps to the entry ```DIMENSION``` in the TSP format.
    #[getset(get_copy = "pub")]
    dim: usize,
    /// The truck capacity for CVRP.
    ///
    /// Maps to the entry ```CAPACITY``` in the TSP format.
    #[getset(get_copy = "pub")]
    capacity: f64,
    /// Specifier for how the edge weights are calculated.
    ///
    /// Maps to the entry ```EDGE_WEIGHT_TYPE``` in the TSP format.
    #[getset(get_copy = "pub")]
    weight_kind: WeightKind,
    /// Specifier for how the edge weights are stored in a file.
    ///
    /// Maps to the entry ```EDGE_WEIGHT_FORMAT``` in the TSP format.
    #[getset(get_copy = "pub")]
    weight_format: WeightFormat,
    /// Specifier for how the edges are stored in a file.
    ///
    /// Maps to the entry ```EDGE_DATA_FORMAT``` in the TSP format.
    #[getset(get = "pub")]
    edge_format: EdgeFormat,
    /// Specifier for how the node coordinates are stored in a file.
    ///
    /// Maps to the entry ```NODE_COORD_TYPE``` in the TSP format.
    #[getset(get_copy = "pub")]
    coord_kind: CoordKind,
    /// Specifier for how node coordinates for display purpose are stored in a file.
    ///
    /// Maps to the entry ```DISPLAY_DATA_TYPE``` in the TSP format.
    #[getset(get_copy = "pub")]
    disp_kind: DisplayKind,
    /// Vector of node coordinates, if available.
    ///
    /// Maps to the entry ```NODE_COORD_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    node_coords: HashMap<usize, Point>,
    /// Vector of depot nodes' id, if available.
    ///
    /// Maps to the entry ```DEPOT_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    depots: HashSet<usize>,
    /// Vector of node demands, if available.
    ///
    /// Maps to the entry ```DEMAND_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    demands: HashMap<usize, f64>,
    /// Vector of edges that *must* appear in solutions to the problem.
    ///
    /// Maps to the entry ```FIXED_EDGES_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    fixed_edges: Vec<(usize, usize)>,
    /// A vector of 2D node coordinates for display purpose, if available.
    ///
    /// Maps to the entry ```DISPLAY_DATA_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    disp_coords: Vec<Point>,
    /// Edge weights in a matrix form as stated in ```EDGE_WEIGHT_FORMAT```, if available.
    ///
    /// Maps to the entry ```EDGE_WEIGHT_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    edge_weights: Vec<Vec<f64>>,
    /// A collection of tours (a sequence of nodes).
    ///
    /// Maps to the entry ```TOUR_SECTION``` in the TSP format.
    #[getset(get = "pub", get_mut = "pub")]
    tours: Vec<Vec<usize>>,
}

impl Tsp {
    /// Returns the node data for a zero-based node id.
    ///
    /// This is the checked counterpart to accessing node coordinate storage directly.
    pub fn node(&self, id: usize) -> Option<&Point> {
        self.node_coords.get(&id)
    }

    /// Returns the edge weight between two nodes if both nodes can be resolved.
    ///
    /// This method is non-panicking. It returns `None` when:
    /// - a node id is out of range,
    /// - an explicit-weight matrix is incomplete for the requested entry, or
    /// - the dataset declares `EDGE_WEIGHT_FORMAT: FUNCTION` / an undefined weight format, which
    ///   this crate does not evaluate dynamically.
    pub fn try_weight(&self, a: usize, b: usize) -> Option<f64> {
        match self.weight_kind {
            WeightKind::Explicit => match self.weight_format {
                WeightFormat::Function | WeightFormat::Undefined => None,
                WeightFormat::FullMatrix => {
                    self.edge_weights.get(a).and_then(|row| row.get(b)).copied()
                }
                WeightFormat::UpperRow | WeightFormat::LowerCol => match a.cmp(&b) {
                    std::cmp::Ordering::Less => self
                        .edge_weights
                        .get(a)
                        .and_then(|row| row.get(b.checked_sub(a + 1)?))
                        .copied(),
                    std::cmp::Ordering::Equal => Some(0.0),
                    std::cmp::Ordering::Greater => self
                        .edge_weights
                        .get(b)
                        .and_then(|row| row.get(a.checked_sub(b + 1)?))
                        .copied(),
                },
                WeightFormat::UpperDiagRow | WeightFormat::LowerDiagCol => {
                    if a < b {
                        self.edge_weights
                            .get(a)
                            .and_then(|row| row.get(b - a))
                            .copied()
                    } else {
                        self.edge_weights
                            .get(b)
                            .and_then(|row| row.get(a - b))
                            .copied()
                    }
                }
                WeightFormat::LowerRow | WeightFormat::UpperCol => match a.cmp(&b) {
                    std::cmp::Ordering::Less => self
                        .edge_weights
                        .get(b.checked_sub(1)?)
                        .and_then(|row| row.get(a))
                        .copied(),
                    std::cmp::Ordering::Equal => Some(0.0),
                    std::cmp::Ordering::Greater => self
                        .edge_weights
                        .get(a.checked_sub(1)?)
                        .and_then(|row| row.get(b))
                        .copied(),
                },
                WeightFormat::LowerDiagRow | WeightFormat::UpperDiagCol => {
                    if a < b {
                        self.edge_weights.get(b).and_then(|row| row.get(a)).copied()
                    } else {
                        self.edge_weights.get(a).and_then(|row| row.get(b)).copied()
                    }
                }
            },
            _ => {
                let na = self.node(a)?;
                let nb = self.node(b)?;
                Some(self.weight_kind.cost(na.pos(), nb.pos()))
            }
        }
    }

    /// Returns the edge weight between two nodes.
    ///
    /// # Arguments
    /// * a - index of the first node.
    /// * b - index of the second node.
    ///
    /// # Panics
    ///
    /// Panics when the requested node or explicit-weight entry is not available.
    /// Prefer [`Tsp::try_weight`] for checked access in production call sites.
    pub fn weight(&self, a: usize, b: usize) -> f64 {
        self.try_weight(a, b)
            .unwrap_or_else(|| panic!("Node or edge not found: a={a}, b={b}"))
    }
}

impl Display for Tsp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Spec: {name} {kind} {dim} {wkind} {wformat} {eformat} {ckind} {dkind}\n
            Data: {coord:?} {eweights:?} {dcoord:?} {fedges:?}",
            name = self.name,
            kind = self.kind,
            dim = self.dim,
            wkind = self.weight_kind,
            wformat = self.weight_format,
            eformat = self.edge_format,
            ckind = self.coord_kind,
            dkind = self.disp_kind,
            coord = self.node_coords,
            eweights = self.edge_weights,
            dcoord = self.disp_coords,
            fedges = self.fixed_edges,
        )
    }
}

/// Responsible for constructing an instance of [`Tsp`].
#[derive(Debug, Default)]
pub struct TspBuilder {
    // Spec
    name: Option<String>,
    kind: Option<TspKind>,
    comment: Option<String>,
    dim: Option<usize>,
    capacity: Option<f64>,
    weight_kind: Option<WeightKind>,
    weight_format: Option<WeightFormat>,
    edge_format: Option<EdgeFormat>,
    coord_kind: Option<CoordKind>,
    disp_kind: Option<DisplayKind>,
    // Data
    coords: Option<HashMap<usize, Point>>,
    depots: Option<HashSet<usize>>,
    demands: Option<HashMap<usize, f64>>,
    edge_weights: Option<Vec<Vec<f64>>>,
    disp_coords: Option<Vec<Point>>,
    fixed_edges: Option<Vec<(usize, usize)>>,
    tours: Option<Vec<Vec<usize>>>,
}

impl TspBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses an input string.
    ///
    /// If all entries in the input string are valid, a [`Tsp`] object will be returned. Otherwise,
    /// an error [`ParseTspError`] is returned, containing hints why the parsing fails.
    ///
    /// This function is intended to fail with structured errors rather than panic on malformed
    /// TSPLIB input.
    // Should be in TryFrom once issue 50133 is fixed.
    // See: https://github.com/rust-lang/rust/issues/50133.
    pub fn parse_str<S>(s: S) -> Result<Tsp, ParseTspError>
    where
        S: AsRef<str>,
    {
        let lines = s
            .as_ref()
            .lines()
            .map(std::borrow::ToOwned::to_owned)
            .collect::<Vec<_>>();
        Self::parse_lines(&lines)
    }

    /// Parses the content of a file given from a path.
    ///
    /// If all entries in the input file are valid, a [`Tsp`] object will be returned. Otherwise,
    /// an error [`ParseTspError`] is returned, containing hints why the parsing fails.
    ///
    /// Unsupported TSPLIB features are reported with [`ParseTspError::UnsupportedFeature`].
    // Should be in TryFrom once issue 50133 is fixed.
    // See: https://github.com/rust-lang/rust/issues/50133.
    pub fn parse_path<P>(path: P) -> Result<Tsp, ParseTspError>
    where
        P: AsRef<Path>,
    {
        if path.as_ref().is_dir() {
            return Err(ParseTspError::Other("Path is a directory"));
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let lines = reader.lines().collect::<Result<Vec<_>, _>>()?;
        Self::parse_lines(&lines)
    }

    fn parse_lines(lines: &[String]) -> Result<Tsp, ParseTspError> {
        let mut builder = TspBuilder::new();
        let mut index = 0;

        while let Some(raw_line) = lines.get(index) {
            index += 1;
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with("EOF") {
                break;
            }

            if line.starts_with(K_NAME) {
                builder.name = Some(Self::split_entry(line, K_NAME)?.to_owned());
            } else if line.starts_with(K_TYPE) {
                builder.kind = Some(TspKind::try_from(InputWrapper(Self::split_entry(
                    line, K_TYPE,
                )?))?);
            } else if line.starts_with("COMMENT") {
                let comment = Self::split_entry(line, "COMMENT")?;
                match builder.comment.as_mut() {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(comment);
                    }
                    None => builder.comment = Some(comment.to_owned()),
                }
            } else if line.starts_with(K_DIM) {
                builder.dim = Some(Self::parse_usize(Self::split_entry(line, K_DIM)?, K_DIM)?);
            } else if line.starts_with("CAPACITY") {
                builder.capacity = Some(Self::parse_f64(Self::split_entry(line, K_CAP)?, K_CAP)?);
            } else if line.starts_with(K_WEIGHT_TYPE) {
                let kind =
                    WeightKind::try_from(InputWrapper(Self::split_entry(line, K_WEIGHT_TYPE)?))?;
                builder.weight_kind = Some(kind);
                builder.coord_kind = Some(CoordKind::from(kind));
            } else if line.starts_with(K_WEIGHT_FORMAT) {
                builder.weight_format = Some(WeightFormat::try_from(InputWrapper(
                    Self::split_entry(line, K_WEIGHT_FORMAT)?,
                ))?);
            } else if line.starts_with(K_EDGE_FORMAT) {
                builder.edge_format = Some(EdgeFormat::try_from(InputWrapper(Self::split_entry(
                    line,
                    K_EDGE_FORMAT,
                )?))?);
            } else if line.starts_with(K_NODE_COORD_TYPE) {
                builder.coord_kind = Some(CoordKind::try_from(InputWrapper(Self::split_entry(
                    line,
                    K_NODE_COORD_TYPE,
                )?))?);
            } else if line.starts_with(K_DISP_TYPE) {
                builder.disp_kind = Some(DisplayKind::try_from(InputWrapper(Self::split_entry(
                    line,
                    K_DISP_TYPE,
                )?))?);
            } else if line.starts_with(K_NODE_COORD_SEC) {
                builder.parse_node_coord_section(lines, &mut index)?;
            } else if line.starts_with("DEPOT_SECTION") {
                builder.parse_depot_section(lines, &mut index)?;
            } else if line.starts_with("DEMAND_SECTION") {
                builder.parse_demand_section(lines, &mut index)?;
            } else if line.starts_with("EDGE_DATA_SECTION") {
                builder.parse_edge_data_section(lines, &mut index)?;
            } else if line.starts_with("FIXED_EDGES_SECTION") {
                builder.parse_fixed_edges_section(lines, &mut index)?;
            } else if line.starts_with("DISPLAY_DATA_SECTION") {
                builder.parse_display_data_section(lines, &mut index)?;
            } else if line.starts_with(K_TOUR_SEC) {
                builder.parse_tour_section(lines, &mut index)?;
            } else if line.starts_with(K_EDGE_WEIGHT_SEC) {
                builder.parse_edge_weight_section(lines, &mut index)?;
            } else {
                return Err(ParseTspError::InvalidEntry(String::from(line)));
            }
        }

        builder.build()
    }

    /// Parse the block `NODE_COORD_SECTION`.
    fn split_entry<'a>(line: &'a str, key: &str) -> Result<&'a str, ParseTspError> {
        let (_, value) = line
            .split_once(':')
            .ok_or_else(|| ParseTspError::InvalidEntry(line.to_string()))?;
        let value = value.trim();
        if value.is_empty() {
            return Err(ParseTspError::InvalidInput {
                key: key.to_string(),
                val: String::new(),
            });
        }
        Ok(value)
    }

    fn next_section_line<'a>(
        lines: &'a [String],
        index: &mut usize,
        section: &str,
    ) -> Result<&'a str, ParseTspError> {
        while let Some(line) = lines.get(*index) {
            *index += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("EOF") {
                return Err(ParseTspError::UnexpectedEof {
                    section: section.to_string(),
                });
            }
            return Ok(trimmed);
        }

        Err(ParseTspError::UnexpectedEof {
            section: section.to_string(),
        })
    }

    fn parse_usize(value: &str, key: &str) -> Result<usize, ParseTspError> {
        value
            .parse::<usize>()
            .map_err(|_| ParseTspError::InvalidInput {
                key: key.to_string(),
                val: value.to_string(),
            })
    }

    fn parse_f64(value: &str, key: &str) -> Result<f64, ParseTspError> {
        value
            .parse::<f64>()
            .map_err(|_| ParseTspError::InvalidInput {
                key: key.to_string(),
                val: value.to_string(),
            })
    }

    fn parse_node_coord_tokens(
        tokens: &[&str],
        coord_kind: CoordKind,
    ) -> Result<Point, ParseTspError> {
        match coord_kind {
            CoordKind::Coord2d => {
                if tokens.len() != 3 {
                    return Err(ParseTspError::InvalidInput {
                        key: K_NODE_COORD_SEC.to_string(),
                        val: tokens.join(" "),
                    });
                }
                Ok(Point::new2(
                    Self::parse_usize(tokens[0], K_NODE_COORD_SEC)?,
                    Self::parse_f64(tokens[1], K_NODE_COORD_SEC)?,
                    Self::parse_f64(tokens[2], K_NODE_COORD_SEC)?,
                ))
            }
            CoordKind::Coord3d => {
                if tokens.len() != 4 {
                    return Err(ParseTspError::InvalidInput {
                        key: K_NODE_COORD_SEC.to_string(),
                        val: tokens.join(" "),
                    });
                }
                Ok(Point::new3(
                    Self::parse_usize(tokens[0], K_NODE_COORD_SEC)?,
                    Self::parse_f64(tokens[1], K_NODE_COORD_SEC)?,
                    Self::parse_f64(tokens[2], K_NODE_COORD_SEC)?,
                    Self::parse_f64(tokens[3], K_NODE_COORD_SEC)?,
                ))
            }
            CoordKind::NoCoord | CoordKind::Undefined => Err(ParseTspError::UnsupportedFeature(
                K_NODE_COORD_SEC.to_string(),
            )),
        }
    }

    fn parse_edge_pair(line: &str, key: &str) -> Result<(usize, usize), ParseTspError> {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return Err(ParseTspError::InvalidInput {
                key: key.to_string(),
                val: line.to_string(),
            });
        }
        Ok((
            Self::parse_usize(tokens[0], key)?,
            Self::parse_usize(tokens[1], key)?,
        ))
    }

    /// Parse the block `NODE_COORD_SECTION`.
    fn parse_node_coord_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        self.validate_spec()?;
        let coord_kind = self.coord_kind.unwrap_or(CoordKind::Undefined);

        let mut count = 0;
        let dim = self.dim.unwrap_or(0);
        let mut dta = HashMap::with_capacity(dim);

        while count < dim {
            let line = Self::next_section_line(lines, index, K_NODE_COORD_SEC)?;
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let pt = Self::parse_node_coord_tokens(&tokens, coord_kind)?;
            dta.insert(pt.id, pt);
            count += 1;
        }

        self.coords = Some(dta);

        Ok(())
    }

    /// Parse the block `DEPOT_SECTION`.
    fn parse_depot_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        self.validate_spec()?;

        let mut dta = HashSet::new();

        loop {
            let line = Self::next_section_line(lines, index, "DEPOT_SECTION")?;
            if line.starts_with("-1") {
                break;
            }

            dta.insert(Self::parse_usize(line, "DEPOT_SECTION")?);
        }

        self.depots = Some(dta);

        Ok(())
    }

    /// Parse the block `DEMAND_SECTION`.
    fn parse_demand_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        self.validate_spec()?;

        let mut dta = HashMap::new();

        for _ in 0..self.dim.unwrap_or(0) {
            let line = Self::next_section_line(lines, index, "DEMAND_SECTION")?;
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            if tokens.len() != 2 {
                return Err(ParseTspError::InvalidInput {
                    key: "DEMAND_SECTION".to_string(),
                    val: line.to_string(),
                });
            }
            dta.insert(
                Self::parse_usize(tokens[0], "DEMAND_SECTION")?,
                Self::parse_f64(tokens[1], "DEMAND_SECTION")?,
            );
        }

        self.demands = Some(dta);

        Ok(())
    }

    /// Parses the ```EDGE_DATA_SECTION```.
    fn parse_edge_data_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        let mut dta = Vec::new();

        match self
            .edge_format
            .as_mut()
            .ok_or_else(|| ParseTspError::MissingEntry(String::from(K_EDGE_FORMAT)))?
        {
            EdgeFormat::EdgeList(v) => {
                loop {
                    let line = Self::next_section_line(lines, index, "EDGE_DATA_SECTION")?;
                    if line.starts_with("-1") {
                        break;
                    }

                    dta.push(Self::parse_edge_pair(line, "EDGE_DATA_SECTION")?);
                }

                v.append(&mut dta);
            }
            EdgeFormat::AdjList => {
                return Err(ParseTspError::UnsupportedFeature("ADJ_LIST".to_string()))
            }
            EdgeFormat::Undefined => {
                return Err(ParseTspError::InvalidEntry(String::from(K_EDGE_FORMAT)))
            }
        }

        Ok(())
    }

    fn parse_fixed_edges_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        let mut dta = Vec::new();

        loop {
            let line = Self::next_section_line(lines, index, "FIXED_EDGES_SECTION")?;
            if line.starts_with("-1") {
                break;
            }

            dta.push(Self::parse_edge_pair(line, "FIXED_EDGES_SECTION")?);
        }

        self.fixed_edges = Some(dta);

        Ok(())
    }

    /// Parses ```TOUR_SECTION```.
    fn parse_tour_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        self.validate_spec()?;
        let mut dta = Vec::new();
        let mut current = Vec::new();

        while let Some(line) = lines.get(*index) {
            *index += 1;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with("EOF") {
                break;
            }
            if line.starts_with("-1") {
                if current.is_empty() {
                    break;
                }
                dta.push(std::mem::take(&mut current));
                continue;
            }

            for token in line.split_whitespace() {
                current.push(Self::parse_usize(token, K_TOUR_SEC)?);
            }
        }

        if !current.is_empty() {
            dta.push(current);
        }

        self.tours = Some(dta);

        Ok(())
    }

    /// Parses ```EDGE_WEIGHT_SECTION```.
    fn parse_edge_weight_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        self.validate_spec()?;
        let dim = self.dim.unwrap_or(0);

        let (len_vec, cnt, row_lengths): (usize, usize, Vec<usize>) = match self
            .weight_format
            .unwrap_or(WeightFormat::Undefined)
        {
            WeightFormat::Function => (0, 0, Vec::new()),
            WeightFormat::FullMatrix => (dim, dim * dim, std::iter::repeat_n(dim, dim).collect()),
            WeightFormat::UpperRow | WeightFormat::LowerCol => (
                dim.saturating_sub(1),
                dim * dim.saturating_sub(1) / 2,
                (1..dim).rev().collect(),
            ),
            WeightFormat::LowerRow | WeightFormat::UpperCol => (
                dim.saturating_sub(1),
                dim * dim.saturating_sub(1) / 2,
                (1..dim).collect(),
            ),
            WeightFormat::UpperDiagRow | WeightFormat::LowerDiagCol => {
                (dim, dim * (dim + 1) / 2, (1..=dim).rev().collect())
            }
            WeightFormat::LowerDiagRow | WeightFormat::UpperDiagCol => {
                (dim, dim * (dim + 1) / 2, (1..=dim).collect())
            }
            WeightFormat::Undefined => {
                return Err(ParseTspError::InvalidEntry(String::from(K_WEIGHT_FORMAT)))
            }
        };

        let mut dta = Vec::with_capacity(len_vec);
        let mut v = Vec::with_capacity(cnt);

        while v.len() < cnt {
            let line = Self::next_section_line(lines, index, K_EDGE_WEIGHT_SEC)?;
            let mut tmp = line
                .split_whitespace()
                .map(|s| Self::parse_f64(s, K_EDGE_WEIGHT_SEC))
                .collect::<Result<Vec<_>, _>>()?;

            v.append(&mut tmp);
        }

        // The SOP files from TSPLIB has an extra line containing dimension in this section,
        // which does not follow the specification.
        if v.len() == dim + 1 {
            v.remove(0);
        }

        if v.len() > cnt && v.len() != dim + 1 {
            return Err(ParseTspError::InvalidInput {
                key: K_EDGE_WEIGHT_SEC.to_string(),
                val: format!("expected {cnt} weights, found {}", v.len()),
            });
        }

        for len_row in row_lengths {
            dta.push(v.drain(0..len_row).collect());
        }

        self.edge_weights = Some(dta);

        Ok(())
    }

    fn parse_display_data_section(
        &mut self,
        lines: &[String],
        index: &mut usize,
    ) -> Result<(), ParseTspError> {
        self.validate_spec()?;
        let dim = self.dim.unwrap_or(0);
        let mut dta = Vec::with_capacity(dim);

        let mut count = 0;
        while count < dim {
            let line = Self::next_section_line(lines, index, "DISPLAY_DATA_SECTION")?;
            let v = line.split_whitespace().collect::<Vec<_>>();
            if v.len() != 3 {
                return Err(ParseTspError::InvalidInput {
                    key: "DISPLAY_DATA_SECTION".to_string(),
                    val: line.to_string(),
                });
            }
            dta.push(Point::new2(
                Self::parse_usize(v[0], "DISPLAY_DATA_SECTION")?,
                Self::parse_f64(v[1], "DISPLAY_DATA_SECTION")?,
                Self::parse_f64(v[2], "DISPLAY_DATA_SECTION")?,
            ));

            count += 1;
        }

        self.disp_coords = Some(dta);

        Ok(())
    }

    /// Validates the specification part.
    fn validate_spec(&self) -> Result<(), ParseTspError> {
        if self.name.is_none() {
            return Err(ParseTspError::MissingEntry(String::from(K_NAME)));
        }

        match self.kind {
            Some(kind) => {
                match kind {
                    TspKind::Tsp | TspKind::Atsp | TspKind::Cvrp | TspKind::Sop => {
                        match self.weight_kind {
                            Some(wk) => {
                                if wk == WeightKind::Undefined {
                                    return Err(ParseTspError::InvalidEntry(String::from(
                                        K_WEIGHT_TYPE,
                                    )));
                                }
                            }
                            None => {
                                return Err(ParseTspError::MissingEntry(String::from(
                                    K_WEIGHT_TYPE,
                                )))
                            }
                        }

                        if kind == TspKind::Cvrp && self.capacity.is_none() {
                            return Err(ParseTspError::MissingEntry(String::from(K_CAP)));
                        }
                    }
                    TspKind::Hcp => match self.edge_format {
                        Some(ref ef) => {
                            if ef == &EdgeFormat::Undefined {
                                return Err(ParseTspError::InvalidEntry(String::from(
                                    K_EDGE_FORMAT,
                                )));
                            }
                        }
                        None => {
                            return Err(ParseTspError::MissingEntry(String::from(K_EDGE_FORMAT)))
                        }
                    },
                    TspKind::Tour => {}
                    TspKind::Undefined => {
                        return Err(ParseTspError::InvalidEntry(String::from(K_TYPE)))
                    }
                }

                if kind != TspKind::Tour && self.dim.is_none() {
                    return Err(ParseTspError::MissingEntry(String::from(K_DIM)));
                }
            }
            None => return Err(ParseTspError::MissingEntry(String::from(K_TYPE))),
        }

        Ok(())
    }

    /// Validates the data part.
    fn validate_data(&self) -> Result<(), ParseTspError> {
        let kind = self
            .kind
            .ok_or_else(|| ParseTspError::MissingEntry(String::from(K_TYPE)))?;

        match kind {
            TspKind::Tsp | TspKind::Atsp | TspKind::Cvrp => match self.weight_kind.unwrap() {
                WeightKind::Explicit => {
                    if self.edge_weights.is_none() {
                        return Err(ParseTspError::MissingEntry(String::from(K_EDGE_WEIGHT_SEC)));
                    }
                }
                _ => {
                    if self.coords.is_none() {
                        return Err(ParseTspError::MissingEntry(String::from(K_NODE_COORD_SEC)));
                    }
                }
            },
            TspKind::Sop => {}
            TspKind::Hcp => {}
            TspKind::Tour => {
                if self.tours.is_none() {
                    return Err(ParseTspError::MissingEntry(String::from(K_TOUR_SEC)));
                }
            }
            TspKind::Undefined => {}
        }

        if let Some(weight_kind) = self.weight_kind {
            match weight_kind {
                WeightKind::Explicit => {
                    if self.edge_weights.is_none() {
                        return Err(ParseTspError::MissingEntry(String::from(K_EDGE_WEIGHT_SEC)));
                    }
                }
                _ => {
                    if self.coords.is_none() {
                        return Err(ParseTspError::MissingEntry(String::from(K_NODE_COORD_SEC)));
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates the inputs and constructs a [`Tsp`] object if the validation is successful.
    /// Otherwise, returns an error [`ParseTspError`].
    pub fn build(self) -> Result<Tsp, ParseTspError> {
        self.validate_spec()?;
        self.validate_data()?;

        let tsp = Tsp {
            name: self.name.unwrap(),
            kind: self.kind.unwrap(),
            comment: self.comment.unwrap_or_default(),
            dim: self.dim.unwrap_or(0),
            capacity: self.capacity.unwrap_or(0.),
            weight_kind: self.weight_kind.unwrap_or(WeightKind::Undefined),
            weight_format: self.weight_format.unwrap_or(WeightFormat::Undefined),
            edge_format: self.edge_format.unwrap_or(EdgeFormat::Undefined),
            coord_kind: self.coord_kind.unwrap_or(CoordKind::Undefined),
            disp_kind: self.disp_kind.unwrap_or(DisplayKind::Undefined),
            node_coords: self.coords.unwrap_or_default(),
            demands: self.demands.unwrap_or_default(),
            depots: self.depots.unwrap_or_default(),
            edge_weights: self.edge_weights.unwrap_or_default(),
            disp_coords: self.disp_coords.unwrap_or_default(),
            fixed_edges: self.fixed_edges.unwrap_or_default(),
            tours: self.tours.unwrap_or_default(),
        };

        Ok(tsp)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct InputWrapper<T>(T);

/// Represents a node coordinate.
#[derive(Clone, Debug)]
pub struct Point {
    /// Id of a point.
    id: usize,
    /// Point's coordinates.
    pos: Vec<f64>,
}

impl Point {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn pos(&self) -> &Vec<f64> {
        &self.pos
    }

    pub fn into_value(self) -> (usize, Vec<f64>) {
        (self.id, self.pos)
    }

    /// Constructs a new point.
    pub fn new(id: usize, pos: Vec<f64>) -> Self {
        Self { id, pos }
    }

    pub fn new2(id: usize, x: f64, y: f64) -> Self {
        Self::new(id - 1, vec![x, y])
    }

    pub fn new3(id: usize, x: f64, y: f64, z: f64) -> Self {
        Self::new(id - 1, vec![x, y, z])
    }
}

/// Enum for TSP's variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TspKind {
    /// Represents a symmetric travelling salesman problem.
    Tsp,
    /// Represents an asymmetric travelling salesman problem.
    Atsp,
    /// Represents a sequential ordering problem.
    Sop,
    /// Represents a Hamiltonian cycle problem.
    Hcp,
    /// Represents a capacitated vehicle routing problem.
    Cvrp,
    /// A collection of tours.
    Tour,
    /// The type of problem is not available.
    Undefined,
}

impl_disp_enum!(TspKind);

impl<T> TryFrom<InputWrapper<T>> for TspKind
where
    T: AsRef<str>,
{
    type Error = ParseTspError;

    fn try_from(value: InputWrapper<T>) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "TSP" => Ok(Self::Tsp),
            "ATSP" => Ok(Self::Atsp),
            "SOP" => Ok(Self::Sop),
            "HCP" => Ok(Self::Hcp),
            "CVRP" => Ok(Self::Cvrp),
            "TOUR" => Ok(Self::Tour),
            _ => Err(ParseTspError::InvalidInput {
                key: K_TYPE.to_string(),
                val: value.0.as_ref().to_string(),
            }),
        }
    }
}

impl From<&str> for TspKind {
    fn from(s: &str) -> Self {
        match s {
            "TSP" => Self::Tsp,
            "ATSP" => Self::Atsp,
            "SOP" => Self::Sop,
            "HCP" => Self::Hcp,
            "CVRP" => Self::Cvrp,
            "TOUR" => Self::Tour,
            _ => Self::Undefined,
        }
    }
}

/// An enum for distance functions defined in the ```TSPLIB``` format.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WeightKind {
    /// Weights are explicitly given in the data file.
    Explicit,
    /// Two-dimensional Euclidean distance.
    Euc2d,
    /// Three-dimensional Euclidean distance.
    Euc3d,
    /// Two-dimensional maximum distance.
    Max2d,
    /// Three-dimensional maximum distance.
    Max3d,
    /// Two-dimensional Manhattan distance.
    Man2d,
    /// Three-dimensional Manhattan distance.
    Man3d,
    /// Rounded-up two dimensional Euclidean distance.
    Ceil2d,
    /// Geographical distance.
    Geo,
    /// Special distance function for problems ```att48``` and ```att532```.
    Att,
    /// Special distance function for crystallography problems of version 1.
    Xray1,
    /// Special distance function for crystallography problems of version 2.
    Xray2,
    /// Distance function defined by users.
    Custom,
    /// No distance function is given.
    Undefined,
}

impl_disp_enum!(WeightKind);

impl From<&str> for WeightKind {
    fn from(s: &str) -> Self {
        match s {
            "EXPLICIT" => Self::Explicit,
            "EUC_2D" => Self::Euc2d,
            "EUC_3D" => Self::Euc3d,
            "MAX_2D" => Self::Max2d,
            "MAX_3D" => Self::Max3d,
            "MAN_2D" => Self::Man2d,
            "MAN_3D" => Self::Man3d,
            "CEIL_2D" => Self::Ceil2d,
            "GEO" => Self::Geo,
            "ATT" => Self::Att,
            "XRAY1" => Self::Xray1,
            "XRAY2" => Self::Xray2,
            "SPECIAL" => Self::Custom,
            _ => Self::Undefined,
        }
    }
}

impl<T> TryFrom<InputWrapper<T>> for WeightKind
where
    T: AsRef<str>,
{
    type Error = ParseTspError;

    fn try_from(value: InputWrapper<T>) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "EXPLICIT" => Ok(Self::Explicit),
            "EUC_2D" => Ok(Self::Euc2d),
            "EUC_3D" => Ok(Self::Euc3d),
            "MAX_2D" => Ok(Self::Max2d),
            "MAX_3D" => Ok(Self::Max3d),
            "MAN_2D" => Ok(Self::Man2d),
            "MAN_3D" => Ok(Self::Man3d),
            "CEIL_2D" => Ok(Self::Ceil2d),
            "GEO" => Ok(Self::Geo),
            "ATT" => Ok(Self::Att),
            "XRAY1" => Ok(Self::Xray1),
            "XRAY2" => Ok(Self::Xray2),
            "SPECIAL" => Ok(Self::Custom),
            _ => Err(ParseTspError::InvalidInput {
                key: K_WEIGHT_TYPE.to_string(),
                val: value.0.as_ref().to_string(),
            }),
        }
    }
}

/// Specifies how edge weights are stored in a file.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WeightFormat {
    /// Weights are calculated by the function stated in [WeightKind].
    ///
    /// Corresponds to the value ```FUNCTION``` in TSPLIB.
    Function,
    /// Weights are given in a full matrix.
    ///
    /// Corresponds to the value ```FULL_MATRIX``` in TSPLIB.
    FullMatrix,
    /// Weights are given in an upper triangular matrix, row-wise without diagonal entries.
    ///
    /// Corresponds to the value ```UPPER_ROW``` in TSPLIB.
    UpperRow,
    /// Weights are given in a lower triangular matrix, row-wise without diagonal entries.
    ///
    /// Corresponds to the value ```LOWE_ROW``` in TSPLIB.
    LowerRow,
    /// Weights are given in an upper triangular matrix, row-wise with diagonal entries.
    ///
    /// Corresponds to the value ```UPPER_DIAG_ROW``` in TSPLIB.
    UpperDiagRow,
    /// Weights are given in a lower triangular matrix, row-wise with diagonal entries.
    ///
    /// Corresponds to the value ```LOWER_DIAG_ROW``` in TSPLIB.
    LowerDiagRow,
    /// Weights are given in an upper triangular matrix, col-wise without diagonal entries.
    ///
    /// Corresponds to the value ```UPPER_COL``` in TSPLIB.
    UpperCol,
    /// Weights are given in an lower triangular matrix, col-wise without diagonal entries.
    ///
    /// Corresponds to the value ```LOWER_COL``` in TSPLIB.
    LowerCol,
    /// Weights are given in an upper triangular matrix, col-wise with diagonal entries.
    ///
    /// Corresponds to the value ```UPPER_DIAG_COL``` in TSPLIB.
    UpperDiagCol,
    /// Weights are given in a lower triangular matrix, col-wise with diagonal entries.
    ///
    /// Corresponds to the value ```LOWER_DIAG_COL``` in TSPLIB.
    LowerDiagCol,
    /// No specification how weights are stored.
    Undefined,
}

impl WeightFormat {
    /// Returns the string value in TSPLIB format.
    #[allow(dead_code)]
    pub(crate) fn tsp_str(&self) -> &'static str {
        match self {
            WeightFormat::Function => "FUNCTION",
            WeightFormat::FullMatrix => "FULL_MATRIX",
            WeightFormat::UpperRow => "UPPER_ROW",
            WeightFormat::LowerRow => "LOWER_ROW",
            WeightFormat::UpperDiagRow => "UPPER_DIAG_ROW",
            WeightFormat::LowerDiagRow => "LOWER_DIAG_ROW",
            WeightFormat::UpperCol => "UPPER_COL",
            WeightFormat::LowerCol => "LOWER_COL",
            WeightFormat::UpperDiagCol => "UPPER_DIAG_COL",
            WeightFormat::LowerDiagCol => "LOWER_DIAG_COL",
            WeightFormat::Undefined => "UNDEFINED",
        }
    }
}

impl From<&str> for WeightFormat {
    fn from(s: &str) -> Self {
        match s {
            "FUNCTION" => Self::Function,
            "FULL_MATRIX" => Self::FullMatrix,
            "UPPER_ROW" => Self::UpperRow,
            "LOWER_ROW" => Self::LowerRow,
            "UPPER_DIAG_ROW" => Self::UpperDiagRow,
            "LOWER_DIAG_ROW" => Self::LowerDiagRow,
            "UPPER_COL" => Self::UpperCol,
            "LOWER_COL" => Self::LowerCol,
            "UPPER_DIAG_COL" => Self::UpperDiagCol,
            "LOWER_DIAG_COL" => Self::LowerDiagCol,
            _ => Self::Undefined,
        }
    }
}

impl<T> TryFrom<InputWrapper<T>> for WeightFormat
where
    T: AsRef<str>,
{
    type Error = ParseTspError;

    fn try_from(value: InputWrapper<T>) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "FUNCTION" => Ok(Self::Function),
            "FULL_MATRIX" => Ok(Self::FullMatrix),
            "UPPER_ROW" => Ok(Self::UpperRow),
            "LOWER_ROW" => Ok(Self::LowerRow),
            "UPPER_DIAG_ROW" => Ok(Self::UpperDiagRow),
            "LOWER_DIAG_ROW" => Ok(Self::LowerDiagRow),
            "UPPER_COL" => Ok(Self::UpperCol),
            "LOWER_COL" => Ok(Self::LowerCol),
            "UPPER_DIAG_COL" => Ok(Self::UpperDiagCol),
            "LOWER_DIAG_COL" => Ok(Self::LowerDiagCol),
            _ => Err(ParseTspError::InvalidInput {
                key: K_WEIGHT_FORMAT.to_string(),
                val: value.0.as_ref().to_string(),
            }),
        }
    }
}

impl_disp_enum!(WeightFormat);

/// Specifies how list of edges are stored in a file.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum EdgeFormat {
    EdgeList(Vec<(usize, usize)>),
    AdjList,
    Undefined,
}

impl From<&str> for EdgeFormat {
    fn from(s: &str) -> Self {
        match s {
            "EDGE_LIST" => Self::EdgeList(Vec::new()),
            "ADJ_LIST" => Self::AdjList,
            _ => Self::Undefined,
        }
    }
}

impl<T> TryFrom<InputWrapper<T>> for EdgeFormat
where
    T: AsRef<str>,
{
    type Error = ParseTspError;

    fn try_from(value: InputWrapper<T>) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "EDGE_LIST" => Ok(Self::EdgeList(Vec::new())),
            "ADJ_LIST" => Ok(Self::AdjList),
            _ => Err(ParseTspError::InvalidInput {
                key: K_EDGE_FORMAT.to_string(),
                val: value.0.as_ref().to_string(),
            }),
        }
    }
}

impl_disp_enum!(EdgeFormat);

/// Specifies how node coordinates are stored in a file.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CoordKind {
    /// Two-dimensional coordinates.
    Coord2d,
    /// Three-dimensional coordinates.
    Coord3d,
    /// No coordinates.
    NoCoord,
    /// Type of node coordinates is undefined.
    Undefined,
}

impl From<&str> for CoordKind {
    fn from(s: &str) -> Self {
        match s {
            "TWOD_COORDS" => Self::Coord2d,
            "THREED_COORDS" => Self::Coord3d,
            "NO_COORDS" => Self::NoCoord,
            _ => Self::Undefined,
        }
    }
}

impl<T> TryFrom<InputWrapper<T>> for CoordKind
where
    T: AsRef<str>,
{
    type Error = ParseTspError;

    fn try_from(value: InputWrapper<T>) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "TWOD_COORDS" => Ok(Self::Coord2d),
            "THREED_COORDS" => Ok(Self::Coord3d),
            "NO_COORDS" => Ok(Self::NoCoord),
            _ => Err(ParseTspError::InvalidInput {
                key: K_NODE_COORD_TYPE.to_string(),
                val: value.0.as_ref().to_string(),
            }),
        }
    }
}

impl From<WeightKind> for CoordKind {
    fn from(kind: WeightKind) -> Self {
        match kind {
            WeightKind::Euc2d
            | WeightKind::Max2d
            | WeightKind::Man2d
            | WeightKind::Ceil2d
            | WeightKind::Geo
            | WeightKind::Att => Self::Coord2d,
            WeightKind::Euc3d | WeightKind::Max3d | WeightKind::Man3d => Self::Coord3d,
            _ => Self::Undefined,
        }
    }
}

impl_disp_enum!(CoordKind);

/// Specifies how node coordinates for display purpose are stored in a file.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DisplayKind {
    /// Display coordinates are based on node coordinates.
    DispCoo,
    /// Two-dimensional coordinates are explicitly given.
    Disp2d,
    /// No display.
    NoDisp,
    /// No information about how to display coordinates.
    Undefined,
}

impl From<&str> for DisplayKind {
    fn from(s: &str) -> Self {
        match s {
            "COORD_DISPLAY" => Self::DispCoo,
            "TWOD_DISPLAY" => Self::Disp2d,
            "NO_DISPLAY" => Self::NoDisp,
            _ => Self::Undefined,
        }
    }
}

impl<T> TryFrom<InputWrapper<T>> for DisplayKind
where
    T: AsRef<str>,
{
    type Error = ParseTspError;

    fn try_from(value: InputWrapper<T>) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "COORD_DISPLAY" => Ok(Self::DispCoo),
            "TWOD_DISPLAY" => Ok(Self::Disp2d),
            "NO_DISPLAY" => Ok(Self::NoDisp),
            _ => Err(ParseTspError::InvalidInput {
                key: K_DISP_TYPE.to_string(),
                val: value.0.as_ref().to_string(),
            }),
        }
    }
}

impl_disp_enum!(DisplayKind);
