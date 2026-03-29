//! Authentication provider trait.
//!
//! Games implement [`AuthProvider`] to connect to their auth backend
//! (OAuth, JWT, custom tokens, etc.). The engine uses this to attach
//! auth tokens to service requests without knowing the auth mechanism.

use crate::error::ServiceError;

/// Current authentication state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthState {
    /// Not authenticated.
    Anonymous,
    /// Authentication in progress.
    Authenticating,
    /// Authenticated with a valid session.
    Authenticated,
    /// Authentication expired, needs refresh.
    Expired,
}

/// Information about the authenticated user.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserInfo {
    /// Unique user identifier.
    pub user_id: String,
    /// Display name (may be empty).
    pub display_name: String,
    /// Optional avatar URL.
    pub avatar_url: Option<String>,
}

/// Trait for authentication providers.
///
/// Games implement this to connect to their specific auth backend:
///
/// ```ignore
/// struct SupabaseAuth { /* ... */ }
///
/// impl AuthProvider for SupabaseAuth {
///     fn state(&self) -> AuthState { /* ... */ }
///     fn token(&self) -> Option<&str> { /* ... */ }
///     // ...
/// }
/// ```
pub trait AuthProvider: Send + Sync {
    /// Current authentication state.
    fn state(&self) -> AuthState;

    /// The current auth token (JWT, session token, etc.).
    /// Returns `None` if not authenticated.
    fn token(&self) -> Option<&str>;

    /// Information about the authenticated user.
    /// Returns `None` if not authenticated.
    fn user_info(&self) -> Option<&UserInfo>;

    /// Attempt to authenticate with credentials.
    /// The credential format is provider-specific (email+password, OAuth code, etc.).
    fn authenticate(&mut self, credentials: &str) -> Result<(), ServiceError>;

    /// Refresh an expired token.
    fn refresh(&mut self) -> Result<(), ServiceError>;

    /// Sign out and clear the session.
    fn sign_out(&mut self);
}

/// A no-op auth provider for games that don't need authentication.
pub struct NoAuth;

impl AuthProvider for NoAuth {
    fn state(&self) -> AuthState {
        AuthState::Anonymous
    }

    fn token(&self) -> Option<&str> {
        None
    }

    fn user_info(&self) -> Option<&UserInfo> {
        None
    }

    fn authenticate(&mut self, _credentials: &str) -> Result<(), ServiceError> {
        Err(ServiceError::Auth(
            "NoAuth provider cannot authenticate".into(),
        ))
    }

    fn refresh(&mut self) -> Result<(), ServiceError> {
        Err(ServiceError::Auth("NoAuth provider cannot refresh".into()))
    }

    fn sign_out(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_auth_state_is_anonymous() {
        let auth = NoAuth;
        assert_eq!(auth.state(), AuthState::Anonymous);
    }

    #[test]
    fn no_auth_token_is_none() {
        let auth = NoAuth;
        assert!(auth.token().is_none());
    }

    #[test]
    fn no_auth_user_info_is_none() {
        let auth = NoAuth;
        assert!(auth.user_info().is_none());
    }

    #[test]
    fn no_auth_authenticate_fails() {
        let mut auth = NoAuth;
        let result = auth.authenticate("some_credentials");
        assert!(result.is_err());
    }

    #[test]
    fn no_auth_refresh_fails() {
        let mut auth = NoAuth;
        let result = auth.refresh();
        assert!(result.is_err());
    }

    #[test]
    fn no_auth_sign_out_does_not_panic() {
        let mut auth = NoAuth;
        auth.sign_out();
    }
}
