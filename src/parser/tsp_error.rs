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
            Self::IoError(e) => write!(f, "{}", format!("IO error: {}", e.to_string())),
            Self::MissingEntry(e) => write!(f, "{}", format!("Missing entry: {}", e)),
            Self::InvalidEntry(e) => write!(f, "{}", format!("Invalid entry: {}", e)),
            Self::InvalidInput { key, val } => {
                write!(f, "{}", format!("Invalid input {} : {}", key, val))
            }
            Self::Other(e) => write!(f, "{}", format!("Invalid entry: {}", e)),
        }
    }
}