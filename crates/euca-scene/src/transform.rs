use std::any::Any;
use euca_math::Transform;
use euca_reflect::Reflect;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalTransform(pub Transform);

impl Default for LocalTransform {
    fn default() -> Self { Self(Transform::IDENTITY) }
}

impl Reflect for LocalTransform {
    fn type_name(&self) -> &'static str { "LocalTransform" }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("translation", format!("({:.3}, {:.3}, {:.3})", self.0.translation.x, self.0.translation.y, self.0.translation.z)),
            ("scale", format!("({:.3}, {:.3}, {:.3})", self.0.scale.x, self.0.scale.y, self.0.scale.z)),
        ]
    }
    fn field_ref(&self, name: &str) -> Option<&dyn Reflect> { match name { "translation" => Some(&self.0.translation), "rotation" => Some(&self.0.rotation), "scale" => Some(&self.0.scale), _ => None } }
    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect> { match name { "translation" => Some(&mut self.0.translation), "rotation" => Some(&mut self.0.rotation), "scale" => Some(&mut self.0.scale), _ => None } }
    fn type_info(&self) -> euca_reflect::TypeInfo { euca_reflect::TypeInfo { name: "LocalTransform", fields: vec![euca_reflect::FieldInfo{name:"translation",type_name:"Vec3"}, euca_reflect::FieldInfo{name:"rotation",type_name:"Quat"}, euca_reflect::FieldInfo{name:"scale",type_name:"Vec3"}] } }
    fn clone_reflect(&self) -> Box<dyn Reflect> { Box::new(*self) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlobalTransform(pub Transform);

impl Default for GlobalTransform {
    fn default() -> Self { Self(Transform::IDENTITY) }
}

impl Reflect for GlobalTransform {
    fn type_name(&self) -> &'static str { "GlobalTransform" }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("world_pos", format!("({:.3}, {:.3}, {:.3})", self.0.translation.x, self.0.translation.y, self.0.translation.z))]
    }
    fn field_ref(&self, name: &str) -> Option<&dyn Reflect> { match name { "translation" | "world_pos" => Some(&self.0.translation), "rotation" => Some(&self.0.rotation), "scale" => Some(&self.0.scale), _ => None } }
    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect> { match name { "translation" | "world_pos" => Some(&mut self.0.translation), "rotation" => Some(&mut self.0.rotation), "scale" => Some(&mut self.0.scale), _ => None } }
    fn type_info(&self) -> euca_reflect::TypeInfo { euca_reflect::TypeInfo { name: "GlobalTransform", fields: vec![euca_reflect::FieldInfo{name:"world_pos",type_name:"Vec3"}, euca_reflect::FieldInfo{name:"rotation",type_name:"Quat"}, euca_reflect::FieldInfo{name:"scale",type_name:"Vec3"}] } }
    fn clone_reflect(&self) -> Box<dyn Reflect> { Box::new(*self) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}
