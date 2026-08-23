use thiserror::Error;

#[derive(Debug, Error)]
pub enum SfmError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error at {context}: {message}")]
    Parse { context: String, message: String },
    #[error("unknown camera model: {0}")]
    UnknownCameraModel(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SfmError>;
