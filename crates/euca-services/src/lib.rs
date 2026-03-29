//! Service integration layer for the Euca engine.
//!
//! Games connect to external backend services (auth, matchmaking, databases)
//! through these trait abstractions. The engine provides the interface;
//! games provide the implementation for their specific backend.
//!
//! # Design Philosophy
//!
//! The engine is a **runtime** — it handles rendering, physics, ECS, and
//! gameplay. Backend services (auth, database, payment, analytics) are
//! **external** concerns that change independently from the engine.
//! This crate provides the bridge between the two worlds.

pub mod auth;
pub mod error;
pub mod http;
pub mod realtime;
pub mod session;
