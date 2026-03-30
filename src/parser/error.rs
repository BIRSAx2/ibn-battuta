use std::fmt::Display;

/// An enum for errors that might occur during parsing.
#[derive(Debug)]
pub enum ParseTspError {
    /// An error due to I/O operations.
    IoError(std::io::Error),
    /// A required entry is missing.
    MissingEntry(String),
    /// A line contains unrecognised keywords.
    InvalidEntry(String),
    /// An entry contains invalid inputs.
    InvalidInput { key: String, val: String },
    /// Parsing stopped before a required section finished.
    UnexpectedEof { section: String },
    /// The input uses a feature that this parser does not support.
    UnsupportedFeature(String),
    /// Any I/O or parsing errors that are not part of this list.
    Other(&'static str),
}

impl From<std::io::Error> for ParseTspError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl Display for ParseTspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::MissingEntry(e) => write!(f, "Missing entry: {e}"),
            Self::InvalidEntry(e) => write!(f, "Invalid entry: {e}"),
            Self::InvalidInput { key, val } => {
                write!(f, "Invalid input {key}: {val}")
            }
            Self::UnexpectedEof { section } => {
                write!(f, "Unexpected end of input while parsing {section}")
            }
            Self::UnsupportedFeature(feature) => {
                write!(f, "Unsupported TSPLIB feature: {feature}")
            }
            Self::Other(e) => write!(f, "Invalid entry: {e}"),
        }
    }
}

impl std::error::Error for ParseTspError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(error) => Some(error),
            _ => None,
        }
    }
}
