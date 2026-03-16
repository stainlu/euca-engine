use std::io;
use std::net::{SocketAddr, UdpSocket};

/// Maximum safe UDP payload (MTU 1500 - IP header 20 - UDP header 8).
pub const MAX_PACKET_SIZE: usize = 1472;

/// Packet header prepended to every UDP message.
///
/// Layout (12 bytes):
/// - sequence: u32 — packet sequence number (wraps)
/// - ack: u32 — last received sequence from remote
/// - ack_bits: u32 — bitfield of previous 32 acks (bit 0 = ack-1, bit 1 = ack-2, ...)
#[derive(Clone, Copy, Debug)]
pub struct PacketHeader {
    pub sequence: u32,
    pub ack: u32,
    pub ack_bits: u32,
}

impl PacketHeader {
    pub const SIZE: usize = 12;

    pub fn write(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.sequence.to_le_bytes());
        buf[4..8].copy_from_slice(&self.ack.to_le_bytes());
        buf[8..12].copy_from_slice(&self.ack_bits.to_le_bytes());
    }

    pub fn read(buf: &[u8]) -> Self {
        Self {
            sequence: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            ack: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            ack_bits: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        }
    }
}

/// Raw UDP transport. Non-blocking socket with send/recv.
pub struct UdpTransport {
    socket: UdpSocket,
    recv_buf: [u8; MAX_PACKET_SIZE],
}

impl UdpTransport {
    /// Bind to a local address.
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            recv_buf: [0u8; MAX_PACKET_SIZE],
        })
    }

    /// Send raw bytes to an address.
    pub fn send_to(&self, data: &[u8], addr: SocketAddr) -> io::Result<usize> {
        self.socket.send_to(data, addr)
    }

    /// Send a packet with header + payload.
    pub fn send_packet(
        &self,
        header: &PacketHeader,
        payload: &[u8],
        addr: SocketAddr,
    ) -> io::Result<()> {
        let total = PacketHeader::SIZE + payload.len();
        assert!(total <= MAX_PACKET_SIZE, "Packet too large: {total}");

        let mut buf = vec![0u8; total];
        header.write(&mut buf[..PacketHeader::SIZE]);
        buf[PacketHeader::SIZE..].copy_from_slice(payload);
        self.socket.send_to(&buf, addr)?;
        Ok(())
    }

    /// Try to receive a packet. Returns None if no data available (non-blocking).
    pub fn recv_packet(&mut self) -> Option<(SocketAddr, PacketHeader, Vec<u8>)> {
        match self.socket.recv_from(&mut self.recv_buf) {
            Ok((len, addr)) => {
                if len < PacketHeader::SIZE {
                    return None; // Too short
                }
                let header = PacketHeader::read(&self.recv_buf[..PacketHeader::SIZE]);
                let payload = self.recv_buf[PacketHeader::SIZE..len].to_vec();
                Some((addr, header, payload))
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => None,
            Err(_) => None,
        }
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_header_roundtrip() {
        let header = PacketHeader {
            sequence: 42,
            ack: 41,
            ack_bits: 0xFFFF_FFFE,
        };
        let mut buf = [0u8; PacketHeader::SIZE];
        header.write(&mut buf);
        let decoded = PacketHeader::read(&buf);
        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.ack, 41);
        assert_eq!(decoded.ack_bits, 0xFFFF_FFFE);
    }

    #[test]
    fn udp_send_recv() {
        let server = UdpTransport::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let client = UdpTransport::bind("127.0.0.1:0".parse().unwrap()).unwrap();

        let server_addr = server.local_addr().unwrap();
        let header = PacketHeader {
            sequence: 1,
            ack: 0,
            ack_bits: 0,
        };
        let payload = b"hello";

        client.send_packet(&header, payload, server_addr).unwrap();

        // Small delay for UDP delivery
        std::thread::sleep(std::time::Duration::from_millis(10));

        let mut server = server;
        let result = server.recv_packet();
        assert!(result.is_some(), "Should receive packet");
        let (_addr, recv_header, recv_payload) = result.unwrap();
        assert_eq!(recv_header.sequence, 1);
        assert_eq!(recv_payload, b"hello");
    }
}
