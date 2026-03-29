//! Texas Hold'em poker game logic.
//!
//! Pure game logic — no rendering, no networking. Can be used by both
//! the server (authoritative) and client (display + prediction).

pub mod betting;
pub mod card;
pub mod deck;
pub mod hand;
pub mod table;
