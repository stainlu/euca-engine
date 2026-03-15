// Nova Reflect - Runtime reflection system
// Re-exports derive macros and provides reflection traits

pub use euca_reflect_derive::*;

/// Trait for types that support runtime reflection.
/// Provides field access by name and type information.
pub trait Reflect: 'static {
    /// Returns the type name.
    fn type_name(&self) -> &'static str;

    /// Returns field names and their string representations.
    fn fields(&self) -> Vec<(&'static str, String)>;
}
