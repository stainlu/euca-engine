//! Common error types for service operations.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("network error: {0}")]
    Network(String),
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("request failed with status {status}: {message}")]
    Http { status: u16, message: String },
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("timeout after {0}ms")]
    Timeout(u64),
    #[error("service unavailable: {0}")]
    Unavailable(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_error_display() {
        let err = ServiceError::Network("connection refused".into());
        assert_eq!(err.to_string(), "network error: connection refused");
    }

    #[test]
    fn auth_error_display() {
        let err = ServiceError::Auth("invalid token".into());
        assert_eq!(err.to_string(), "authentication failed: invalid token");
    }

    #[test]
    fn http_error_display() {
        let err = ServiceError::Http {
            status: 404,
            message: "not found".into(),
        };
        assert_eq!(err.to_string(), "request failed with status 404: not found");
    }

    #[test]
    fn serialization_error_display() {
        let err = ServiceError::Serialization("invalid json".into());
        assert_eq!(err.to_string(), "serialization error: invalid json");
    }

    #[test]
    fn timeout_error_display() {
        let err = ServiceError::Timeout(5000);
        assert_eq!(err.to_string(), "timeout after 5000ms");
    }

    #[test]
    fn unavailable_error_display() {
        let err = ServiceError::Unavailable("maintenance".into());
        assert_eq!(err.to_string(), "service unavailable: maintenance");
    }
}
