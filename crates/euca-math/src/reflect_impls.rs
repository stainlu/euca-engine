//! Reflect impls for math types (enabled by the reflect feature).
use euca_reflect::{FieldInfo, Reflect, TypeInfo};
use std::any::Any;
use crate::{Mat4, Quat, Vec3, Vec4};

macro_rules! impl_reflect_vec {
    ($ty:ident, $name:expr, [$($field:ident),+]) => {
        impl Reflect for $ty {
            fn type_name(&self) -> &'static str { $name }
            fn fields(&self) -> Vec<(&'static str, String)> { vec![$(( stringify!($field), format!("{:?}", self.$field) )),+] }
            fn field_ref(&self, name: &str) -> Option<&dyn Reflect> { match name { $( stringify!($field) => Some(&self.$field), )+ _ => None } }
            fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect> { match name { $( stringify!($field) => Some(&mut self.$field), )+ _ => None } }
            fn set_field(&mut self, name: &str, value: &dyn Reflect) -> bool {
                if let Some(v) = value.as_any().downcast_ref::<f32>() { match name { $( stringify!($field) => { self.$field = *v; true } )+ _ => false } } else { false }
            }
            fn type_info(&self) -> TypeInfo { TypeInfo { name: $name, fields: vec![$( FieldInfo { name: stringify!($field), type_name: "f32" }, )+] } }
            fn clone_reflect(&self) -> Box<dyn Reflect> { Box::new(*self) }
            fn as_any(&self) -> &dyn Any { self }
            fn as_any_mut(&mut self) -> &mut dyn Any { self }
        }
    };
}
impl_reflect_vec!(Vec3, "Vec3", [x, y, z]);
impl_reflect_vec!(Vec4, "Vec4", [x, y, z, w]);
impl_reflect_vec!(Quat, "Quat", [x, y, z, w]);

impl Reflect for Mat4 {
    fn type_name(&self) -> &'static str { "Mat4" }
    fn fields(&self) -> Vec<(&'static str, String)> { vec![("cols", format!("{:?}", self.cols))] }
    fn type_info(&self) -> TypeInfo { TypeInfo { name: "Mat4", fields: vec![FieldInfo { name: "cols", type_name: "[[f32; 4]; 4]" }] } }
    fn clone_reflect(&self) -> Box<dyn Reflect> { Box::new(*self) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}
