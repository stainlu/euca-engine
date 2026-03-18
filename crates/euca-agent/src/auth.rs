//! nit-compatible agent authentication.
//!
//! Agents authenticate via Ed25519 signatures produced by `nit sign --login`.
//! The login payload includes `{agent_id, domain, timestamp, signature, public_key}`.
//! The server verifies the signature against the public key and issues a session token.
//! No external HTTP calls — works fully offline.

use base64::Engine;
use ed25519_dalek::{Signature, VerifyingKey};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Login payload from the agent (produced by `nit sign --login <domain>`).
#[derive(Deserialize)]
pub struct LoginPayload {
    pub agent_id: String,
    pub domain: String,
    pub timestamp: u64,
    pub signature: String,
    /// Ed25519 public key, base64-encoded (32 bytes).
    pub public_key: String,
}

/// Successful login response.
#[derive(Serialize)]
pub struct LoginResponse {
    pub ok: bool,
    pub session_token: String,
    pub agent_id: String,
}

/// Error response.
#[derive(Serialize)]
pub struct AuthError {
    pub ok: bool,
    pub error: String,
}

/// Active session.
struct Session {
    agent_id: String,
    #[allow(dead_code)]
    created_at: u64,
}

/// Session store for authenticated agents.
#[derive(Clone)]
pub struct AuthStore {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl AuthStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Verify a nit login payload and create a session.
    pub fn login(&self, payload: &LoginPayload) -> Result<String, String> {
        // Decode public key
        let pk_bytes = base64::engine::general_purpose::STANDARD
            .decode(&payload.public_key)
            .map_err(|e| format!("Invalid public_key base64: {e}"))?;

        let pk_array: [u8; 32] = pk_bytes
            .try_into()
            .map_err(|_| "Public key must be 32 bytes".to_string())?;

        let verifying_key = VerifyingKey::from_bytes(&pk_array)
            .map_err(|e| format!("Invalid Ed25519 public key: {e}"))?;

        // Decode signature
        let sig_bytes = base64::engine::general_purpose::STANDARD
            .decode(&payload.signature)
            .map_err(|e| format!("Invalid signature base64: {e}"))?;

        let sig_array: [u8; 64] = sig_bytes
            .try_into()
            .map_err(|_| "Signature must be 64 bytes".to_string())?;

        let signature = Signature::from_bytes(&sig_array);

        // Reconstruct the signed message: "{agent_id}\n{domain}\n{timestamp}"
        let message = format!("{}\n{}\n{}", payload.agent_id, payload.domain, payload.timestamp);

        // Verify signature
        use ed25519_dalek::Verifier;
        verifying_key
            .verify(message.as_bytes(), &signature)
            .map_err(|_| "Signature verification failed".to_string())?;

        // Check timestamp is within 5 minutes
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if now.abs_diff(payload.timestamp) > 300 {
            return Err("Timestamp expired (>5 minutes)".to_string());
        }

        // Generate session token
        let token: String = {
            let mut rng = rand::thread_rng();
            (0..32)
                .map(|_| format!("{:02x}", rng.r#gen::<u8>()))
                .collect()
        };

        // Store session
        self.sessions.write().unwrap().insert(
            token.clone(),
            Session {
                agent_id: payload.agent_id.clone(),
                created_at: now,
            },
        );

        log::info!("Agent {} authenticated", payload.agent_id);
        Ok(token)
    }

    /// Look up agent_id for a session token.
    pub fn validate(&self, token: &str) -> Option<String> {
        self.sessions
            .read()
            .unwrap()
            .get(token)
            .map(|s| s.agent_id.clone())
    }
}

impl Default for AuthStore {
    fn default() -> Self {
        Self::new()
    }
}
