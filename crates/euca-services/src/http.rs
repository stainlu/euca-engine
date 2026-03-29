//! Platform-agnostic HTTP client trait.
//!
//! Abstracts HTTP requests so the engine (and games) can make API calls
//! that work on both native (reqwest/tokio) and WASM (fetch API).

use crate::error::ServiceError;
use std::collections::HashMap;

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

/// An HTTP request.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
}

/// An HTTP response.
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Response {
    /// Parse the body as UTF-8 text.
    pub fn text(&self) -> Result<&str, ServiceError> {
        std::str::from_utf8(&self.body).map_err(|e| ServiceError::Serialization(e.to_string()))
    }

    /// Parse the body as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, ServiceError> {
        serde_json::from_slice(&self.body).map_err(|e| ServiceError::Serialization(e.to_string()))
    }

    /// Whether the status code indicates success (2xx).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Platform-agnostic HTTP client.
///
/// On native, typically backed by reqwest. On WASM, backed by fetch API.
/// Games can also implement custom clients for testing or specialized needs.
pub trait HttpClient: Send + Sync {
    /// Execute an HTTP request synchronously (blocking on native, immediate on WASM).
    fn execute(&self, request: Request) -> Result<Response, ServiceError>;
}

impl Request {
    /// Create a GET request.
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: Method::Get,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
        }
    }

    /// Create a POST request with JSON body.
    pub fn post_json<T: serde::Serialize>(
        url: impl Into<String>,
        body: &T,
    ) -> Result<Self, ServiceError> {
        let json =
            serde_json::to_vec(body).map_err(|e| ServiceError::Serialization(e.to_string()))?;
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        Ok(Self {
            method: Method::Post,
            url: url.into(),
            headers,
            body: Some(json),
        })
    }

    /// Add an authorization header with a bearer token.
    pub fn with_bearer_token(mut self, token: &str) -> Self {
        self.headers
            .insert("Authorization".to_string(), format!("Bearer {token}"));
        self
    }

    /// Add a custom header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_get_builds_correctly() {
        let req = Request::get("https://api.example.com/users");
        assert_eq!(req.method, Method::Get);
        assert_eq!(req.url, "https://api.example.com/users");
        assert!(req.headers.is_empty());
        assert!(req.body.is_none());
    }

    #[test]
    fn request_post_json_serializes_body() {
        #[derive(serde::Serialize)]
        struct Payload {
            name: String,
        }
        let req = Request::post_json(
            "https://api.example.com/users",
            &Payload {
                name: "Alice".into(),
            },
        )
        .unwrap();
        assert_eq!(req.method, Method::Post);
        assert_eq!(req.headers.get("Content-Type").unwrap(), "application/json");
        assert!(req.body.is_some());

        let body: serde_json::Value = serde_json::from_slice(&req.body.unwrap()).unwrap();
        assert_eq!(body["name"], "Alice");
    }

    #[test]
    fn request_with_bearer_token() {
        let req = Request::get("https://api.example.com").with_bearer_token("my-jwt");
        assert_eq!(req.headers.get("Authorization").unwrap(), "Bearer my-jwt");
    }

    #[test]
    fn request_with_header() {
        let req = Request::get("https://api.example.com").with_header("X-Custom", "custom-value");
        assert_eq!(req.headers.get("X-Custom").unwrap(), "custom-value");
    }

    #[test]
    fn response_is_success_for_2xx() {
        for status in 200..300u16 {
            let resp = Response {
                status,
                headers: HashMap::new(),
                body: Vec::new(),
            };
            assert!(resp.is_success(), "status {status} should be success");
        }
    }

    #[test]
    fn response_is_not_success_outside_2xx() {
        for status in [100, 199, 300, 404, 500] {
            let resp = Response {
                status,
                headers: HashMap::new(),
                body: Vec::new(),
            };
            assert!(!resp.is_success(), "status {status} should not be success");
        }
    }

    #[test]
    fn response_text_parses_utf8() {
        let resp = Response {
            status: 200,
            headers: HashMap::new(),
            body: b"hello world".to_vec(),
        };
        assert_eq!(resp.text().unwrap(), "hello world");
    }

    #[test]
    fn response_text_rejects_invalid_utf8() {
        let resp = Response {
            status: 200,
            headers: HashMap::new(),
            body: vec![0xFF, 0xFE],
        };
        assert!(resp.text().is_err());
    }

    #[test]
    fn response_json_parses_valid_json() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct User {
            id: u32,
            name: String,
        }
        let resp = Response {
            status: 200,
            headers: HashMap::new(),
            body: br#"{"id":42,"name":"Bob"}"#.to_vec(),
        };
        let user: User = resp.json().unwrap();
        assert_eq!(
            user,
            User {
                id: 42,
                name: "Bob".into()
            }
        );
    }

    #[test]
    fn response_json_rejects_invalid_json() {
        let resp = Response {
            status: 200,
            headers: HashMap::new(),
            body: b"not json".to_vec(),
        };
        let result: Result<serde_json::Value, _> = resp.json();
        assert!(result.is_err());
    }
}
