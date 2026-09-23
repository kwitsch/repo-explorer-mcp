//! In-process candle judge backend (feature `candle`). A faithful port of
//! Laya 0.3.7 inference for the one fixed judge question.

pub mod calib;
pub(crate) mod checkpoint;
pub(crate) mod head;
mod sequence;

pub use sequence::{Encoded, SpecialIds, build_sequence};
