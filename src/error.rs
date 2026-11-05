use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsError {
    InvalidHandle,
    InvalidFrame,
    FrameTooDeep,
    NonFinite(&'static str),
    InvalidShape,
    InvalidMaterial,
    InvalidSettings,
    NotDynamic,
    OutOfRange,
    DispatchIncomplete,
    InvalidJoint,
    BodyHasJoints,
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle => f.write_str("invalid body handle"),
            Self::InvalidFrame => f.write_str("invalid frame"),
            Self::FrameTooDeep => f.write_str("frame hierarchy too deep"),
            Self::NonFinite(what) => write!(f, "non finite value in {what}"),
            Self::InvalidShape => f.write_str("invalid shape"),
            Self::InvalidMaterial => f.write_str("invalid material"),
            Self::InvalidSettings => f.write_str("invalid settings"),
            Self::NotDynamic => f.write_str("body is not dynamic"),
            Self::OutOfRange => f.write_str("position out of range"),
            Self::DispatchIncomplete => f.write_str("dispatcher did not run every job"),
            Self::InvalidJoint => f.write_str("invalid joint"),
            Self::BodyHasJoints => f.write_str("body has joints"),
        }
    }
}

impl Error for PhysicsError {}
