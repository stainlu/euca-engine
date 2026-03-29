//! Minimal web hello-world for the Euca engine.
//!
//! Build with `wasm-pack build --target web` (or `trunk build`), then serve
//! `index.html` from a local HTTP server.
//!
//! On native targets this crate is an empty cdylib — all logic is wasm32-only.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Entry point called automatically when the WASM module is instantiated.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    euca_web::init_wasm();
    euca_web::run_web(euca_web::WebConfig::new("#game-canvas", 1280, 720));
}
