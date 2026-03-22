//! QUIC transport using Quinn — encrypted, congestion-controlled connections.
//!
//! Provides TLS 1.3 encryption, built-in congestion control (Cubic),
//! connection management, and both reliable streams and unreliable datagrams.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

/// QUIC transport wrapping a Quinn endpoint.
pub struct QuicTransport {
    endpoint: quinn::Endpoint,
    connections: HashMap<SocketAddr, quinn::Connection>,
}

impl QuicTransport {
    /// Create a server endpoint that accepts incoming connections.
    ///
    /// Uses the provided certificate and private key for TLS.
    pub fn server(bind_addr: SocketAddr, cert_der: &[u8], key_der: &[u8]) -> Result<Self, String> {
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(key_der.to_vec());
        let cert = rustls::pki_types::CertificateDer::from(cert_der.to_vec());

        let mut server_crypto = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert], key.into())
            .map_err(|e| format!("TLS server config failed: {e}"))?;
        server_crypto.alpn_protocols = vec![b"euca".to_vec()];

        let server_config = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
                .map_err(|e| format!("QUIC server config failed: {e}"))?,
        ));

        let endpoint = quinn::Endpoint::server(server_config, bind_addr)
            .map_err(|e| format!("Failed to bind QUIC server to {bind_addr}: {e}"))?;

        log::info!("QUIC server listening on {bind_addr}");

        Ok(Self {
            endpoint,
            connections: HashMap::new(),
        })
    }

    /// Create a client endpoint (no listening, only outgoing connections).
    pub fn client() -> Result<Self, String> {
        let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| format!("Failed to create QUIC client: {e}"))?;

        // Accept self-signed certificates for development
        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();

        let mut transport = quinn::TransportConfig::default();
        transport.max_idle_timeout(Some(
            quinn::IdleTimeout::try_from(std::time::Duration::from_secs(30)).unwrap(),
        ));

        let mut client_config = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
                .map_err(|e| format!("QUIC client config failed: {e}"))?,
        ));
        client_config.transport_config(Arc::new(transport));
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            endpoint,
            connections: HashMap::new(),
        })
    }

    /// Connect to a QUIC server.
    pub async fn connect(&mut self, addr: SocketAddr, server_name: &str) -> Result<(), String> {
        let connection = self
            .endpoint
            .connect(addr, server_name)
            .map_err(|e| format!("Connect failed: {e}"))?
            .await
            .map_err(|e| format!("Connection handshake failed: {e}"))?;

        log::info!("QUIC connected to {addr}");
        self.connections.insert(addr, connection);
        Ok(())
    }

    /// Accept the next incoming connection (server-side).
    pub async fn accept(&mut self) -> Option<SocketAddr> {
        let incoming = self.endpoint.accept().await?;
        let connection = incoming.await.ok()?;
        let addr = connection.remote_address();
        log::info!("QUIC accepted connection from {addr}");
        self.connections.insert(addr, connection);
        Some(addr)
    }

    /// Send data reliably to a peer via a unidirectional QUIC stream.
    pub async fn send_reliable(&self, addr: &SocketAddr, data: &[u8]) -> Result<(), String> {
        let conn = self
            .connections
            .get(addr)
            .ok_or_else(|| format!("No connection to {addr}"))?;

        let mut stream = conn
            .open_uni()
            .await
            .map_err(|e| format!("Failed to open stream: {e}"))?;

        // Write length prefix + data
        let len = (data.len() as u32).to_le_bytes();
        stream
            .write_all(&len)
            .await
            .map_err(|e| format!("Write length failed: {e}"))?;
        stream
            .write_all(data)
            .await
            .map_err(|e| format!("Write data failed: {e}"))?;
        stream
            .finish()
            .map_err(|e| format!("Finish stream failed: {e}"))?;

        Ok(())
    }

    /// Send data unreliably via QUIC datagram (for state updates).
    ///
    /// Datagrams may be lost or arrive out of order, similar to UDP.
    /// Use for frequently-updated state where latest value matters more than reliability.
    pub fn send_unreliable(&self, addr: &SocketAddr, data: &[u8]) -> Result<(), String> {
        let conn = self
            .connections
            .get(addr)
            .ok_or_else(|| format!("No connection to {addr}"))?;

        conn.send_datagram(data.to_vec().into())
            .map_err(|e| format!("Datagram send failed: {e}"))?;

        Ok(())
    }

    /// Receive the next datagram from any connected peer.
    pub async fn recv_datagram(&self, addr: &SocketAddr) -> Option<Vec<u8>> {
        let conn = self.connections.get(addr)?;
        let datagram = conn.read_datagram().await.ok()?;
        Some(datagram.to_vec())
    }

    /// Check if a peer is connected.
    pub fn is_connected(&self, addr: &SocketAddr) -> bool {
        self.connections.contains_key(addr)
    }

    /// Disconnect a peer.
    pub fn disconnect(&mut self, addr: &SocketAddr) {
        if let Some(conn) = self.connections.remove(addr) {
            conn.close(quinn::VarInt::from_u32(0), b"disconnect");
            log::info!("Disconnected {addr}");
        }
    }

    /// Number of active connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Get all connected peer addresses.
    pub fn connected_peers(&self) -> Vec<SocketAddr> {
        self.connections.keys().copied().collect()
    }
}

/// Ensure the rustls CryptoProvider is installed (call once before using QUIC).
pub fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Generate a self-signed certificate for development/LAN play.
///
/// Returns `(cert_der, key_der)` as DER-encoded bytes.
pub fn generate_self_signed_cert() -> (Vec<u8>, Vec<u8>) {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])
        .expect("Certificate generation should not fail");
    let cert_der = cert.cert.der().to_vec();
    let key_der = cert.key_pair.serialize_der();
    (cert_der, key_der)
}

/// Skip server certificate verification (for development with self-signed certs).
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_cert_succeeds() {
        ensure_crypto_provider();
        let (cert, key) = generate_self_signed_cert();
        assert!(!cert.is_empty());
        assert!(!key.is_empty());
    }

    #[test]
    fn client_creation_succeeds() {
        ensure_crypto_provider();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _guard = rt.enter();
        let client = QuicTransport::client();
        assert!(client.is_ok(), "Client creation failed: {:?}", client.err());
    }

    #[test]
    fn server_creation_succeeds() {
        ensure_crypto_provider();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _guard = rt.enter();
        let (cert, key) = generate_self_signed_cert();
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = QuicTransport::server(addr, &cert, &key);
        assert!(server.is_ok(), "Server creation failed: {:?}", server.err());
    }
}
