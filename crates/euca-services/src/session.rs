//! Session management — coordinates auth + services for a game session.

use crate::auth::{AuthProvider, AuthState, NoAuth};

/// Manages the current game session, coordinating auth state with
/// service connections. Games create a `Session` at startup and pass
/// it to systems that need service access.
pub struct Session {
    auth: Box<dyn AuthProvider>,
}

impl Session {
    /// Create a session with no authentication.
    pub fn anonymous() -> Self {
        Self {
            auth: Box::new(NoAuth),
        }
    }

    /// Create a session with a custom auth provider.
    pub fn with_auth(auth: impl AuthProvider + 'static) -> Self {
        Self {
            auth: Box::new(auth),
        }
    }

    /// Access the auth provider.
    pub fn auth(&self) -> &dyn AuthProvider {
        &*self.auth
    }

    /// Mutable access to the auth provider.
    pub fn auth_mut(&mut self) -> &mut dyn AuthProvider {
        &mut *self.auth
    }

    /// Whether the session is authenticated.
    pub fn is_authenticated(&self) -> bool {
        self.auth.state() == AuthState::Authenticated
    }

    /// Get the current auth token, if any.
    pub fn token(&self) -> Option<&str> {
        self.auth.token()
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::anonymous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{AuthState, UserInfo};
    use crate::error::ServiceError;

    #[test]
    fn anonymous_session_is_not_authenticated() {
        let session = Session::anonymous();
        assert!(!session.is_authenticated());
        assert!(session.token().is_none());
    }

    #[test]
    fn default_session_is_anonymous() {
        let session = Session::default();
        assert!(!session.is_authenticated());
        assert_eq!(session.auth().state(), AuthState::Anonymous);
    }

    /// A mock auth provider for testing `Session::with_auth`.
    struct MockAuth {
        authenticated: bool,
    }

    impl AuthProvider for MockAuth {
        fn state(&self) -> AuthState {
            if self.authenticated {
                AuthState::Authenticated
            } else {
                AuthState::Anonymous
            }
        }

        fn token(&self) -> Option<&str> {
            if self.authenticated {
                Some("mock-token-123")
            } else {
                None
            }
        }

        fn user_info(&self) -> Option<&UserInfo> {
            None
        }

        fn authenticate(&mut self, _credentials: &str) -> Result<(), ServiceError> {
            self.authenticated = true;
            Ok(())
        }

        fn refresh(&mut self) -> Result<(), ServiceError> {
            Ok(())
        }

        fn sign_out(&mut self) {
            self.authenticated = false;
        }
    }

    #[test]
    fn session_with_auth_delegates_state() {
        let session = Session::with_auth(MockAuth {
            authenticated: true,
        });
        assert!(session.is_authenticated());
        assert_eq!(session.token(), Some("mock-token-123"));
    }

    #[test]
    fn session_with_auth_mutable_access() {
        let mut session = Session::with_auth(MockAuth {
            authenticated: false,
        });
        assert!(!session.is_authenticated());

        session.auth_mut().authenticate("creds").unwrap();
        assert!(session.is_authenticated());
        assert_eq!(session.token(), Some("mock-token-123"));

        session.auth_mut().sign_out();
        assert!(!session.is_authenticated());
    }
}
