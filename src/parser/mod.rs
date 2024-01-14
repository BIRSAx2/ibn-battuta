//! A module for parsing TSPLIB file formats.
pub use tsp::CoordKind;
pub use tsp::DisplayKind;
pub use tsp::EdgeFormat;
pub use tsp::Tsp;
pub use tsp::TspBuilder;
pub use tsp::TspKind;
pub use tsp::WeightFormat;
pub use tsp::WeightKind;
pub use tsp_error::ParseTspError;

/// Macro for implementing trait Display for Enums.
macro_rules! impl_disp_enum {
    ($enm:ident) => {
        impl std::fmt::Display for $enm {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self)
            }
        }
    };
}

mod tsp_error;

pub mod metric;

pub mod tsp;

mod tests;

