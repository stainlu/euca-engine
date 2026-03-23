// Euca Reflect - Runtime reflection system
pub use euca_reflect_derive::*;
use std::any::Any;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FieldInfo {
    pub name: &'static str,
    pub type_name: &'static str,
}
#[derive(Clone, Debug)]
pub struct TypeInfo {
    pub name: &'static str,
    pub fields: Vec<FieldInfo>,
}

pub trait Reflect: 'static + Send + Sync {
    fn type_name(&self) -> &'static str;
    fn fields(&self) -> Vec<(&'static str, String)>;
    fn field_ref(&self, _name: &str) -> Option<&dyn Reflect> {
        None
    }
    fn field_mut(&mut self, _name: &str) -> Option<&mut dyn Reflect> {
        None
    }
    fn set_field(&mut self, _name: &str, _value: &dyn Reflect) -> bool {
        false
    }
    fn type_info(&self) -> TypeInfo {
        TypeInfo {
            name: self.type_name(),
            fields: Vec::new(),
        }
    }
    fn clone_reflect(&self) -> Box<dyn Reflect>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub struct TypeRegistry {
    registrations: HashMap<&'static str, TypeRegistration>,
}
pub struct TypeRegistration {
    pub name: &'static str,
    pub info: TypeInfo,
    factory: Box<dyn Fn() -> Box<dyn Reflect> + Send + Sync>,
}
impl TypeRegistry {
    pub fn new() -> Self {
        Self {
            registrations: HashMap::new(),
        }
    }
    pub fn register<T: Reflect + Default>(&mut self) {
        let s = T::default();
        let n = s.type_name();
        let i = s.type_info();
        self.registrations.insert(
            n,
            TypeRegistration {
                name: n,
                info: i,
                factory: Box::new(|| Box::new(T::default())),
            },
        );
    }
    pub fn get_by_name(&self, name: &str) -> Option<&TypeRegistration> {
        self.registrations.get(name)
    }
    pub fn create_default(&self, name: &str) -> Option<Box<dyn Reflect>> {
        self.registrations.get(name).map(|r| (r.factory)())
    }
}
impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_reflect_primitive {
    ($ty:ty, $name:expr) => {
        impl Reflect for $ty {
            fn type_name(&self) -> &'static str {
                $name
            }
            fn fields(&self) -> Vec<(&'static str, String)> {
                Vec::new()
            }
            fn clone_reflect(&self) -> Box<dyn Reflect> {
                Box::new(self.clone())
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }
    };
}
impl_reflect_primitive!(f32, "f32");
impl_reflect_primitive!(f64, "f64");
impl_reflect_primitive!(i32, "i32");
impl_reflect_primitive!(u32, "u32");
impl_reflect_primitive!(i64, "i64");
impl_reflect_primitive!(u64, "u64");
impl_reflect_primitive!(bool, "bool");
impl_reflect_primitive!(String, "String");

#[cfg(feature = "json")]
pub mod json {
    use super::{Reflect, TypeRegistry};

    /// Attempt to convert a leaf (field-less) [`Reflect`] value into a JSON
    /// primitive by matching on the well-known type name returned by the
    /// `Reflect::type_name` implementation for each built-in primitive.
    fn primitive_to_json(value: &dyn Reflect) -> Result<serde_json::Value, String> {
        let any = value.as_any();
        match value.type_name() {
            "f32" => Ok(any.downcast_ref::<f32>().copied().unwrap().into()),
            "f64" => Ok(any.downcast_ref::<f64>().copied().unwrap().into()),
            "i32" => Ok(any.downcast_ref::<i32>().copied().unwrap().into()),
            "u32" => Ok(any.downcast_ref::<u32>().copied().unwrap().into()),
            "i64" => Ok(any.downcast_ref::<i64>().copied().unwrap().into()),
            "u64" => Ok(any.downcast_ref::<u64>().copied().unwrap().into()),
            "bool" => Ok(any.downcast_ref::<bool>().copied().unwrap().into()),
            "String" => Ok(any.downcast_ref::<String>().cloned().unwrap().into()),
            other => Err(format!("unsupported reflect type: {other}")),
        }
    }

    /// Serialize a [`Reflect`] value to [`serde_json::Value`].
    ///
    /// Leaf types (no fields) are converted via [`primitive_to_json`].
    /// Struct-like types produce a JSON object with a `__type` discriminator.
    ///
    /// Returns `Err` if the value (or any nested field) is a leaf whose type
    /// name is not a recognized primitive.
    pub fn reflect_to_json(value: &dyn Reflect) -> Result<serde_json::Value, String> {
        let fields = value.fields();
        if fields.is_empty() {
            primitive_to_json(value)
        } else {
            let mut map = serde_json::Map::new();
            map.insert("__type".into(), value.type_name().into());
            for (name, repr) in &fields {
                let v = match value.field_ref(name) {
                    Some(fv) => reflect_to_json(fv)?,
                    None => serde_json::Value::String(repr.clone()),
                };
                map.insert(name.to_string(), v);
            }
            Ok(serde_json::Value::Object(map))
        }
    }
    pub fn reflect_from_json(
        json: &serde_json::Value,
        registry: &TypeRegistry,
    ) -> Option<Box<dyn Reflect>> {
        let obj = json.as_object()?;
        let tn = obj.get("__type")?.as_str()?;
        let mut inst = registry.create_default(tn)?;
        for (k, v) in obj {
            if k == "__type" {
                continue;
            }
            if let Some(fv) = json_to_val(v, registry) {
                inst.set_field(k, fv.as_ref());
            }
        }
        Some(inst)
    }
    fn json_to_val(v: &serde_json::Value, reg: &TypeRegistry) -> Option<Box<dyn Reflect>> {
        match v {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    Some(Box::new(f as f32))
                } else if let Some(i) = n.as_i64() {
                    Some(Box::new(i))
                } else {
                    n.as_u64().map(|u| Box::new(u) as Box<dyn Reflect>)
                }
            }
            serde_json::Value::Bool(b) => Some(Box::new(*b)),
            serde_json::Value::String(s) => Some(Box::new(s.clone())),
            serde_json::Value::Object(_) => reflect_from_json(v, reg),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primitive_type_name() {
        assert_eq!(Reflect::type_name(&42_f32), "f32");
        assert_eq!(Reflect::type_name(&true), "bool");
    }
    #[test]
    fn primitive_clone() {
        let c = 3.14_f32.clone_reflect();
        assert_eq!(*c.as_any().downcast_ref::<f32>().unwrap(), 3.14_f32);
    }
    #[test]
    fn primitive_any_mut() {
        let mut v = 10_i32;
        *v.as_any_mut().downcast_mut::<i32>().unwrap() = 42;
        assert_eq!(v, 42);
    }
    #[test]
    fn primitive_no_fields() {
        assert!(42_u32.fields().is_empty());
        assert!(42_u32.field_ref("x").is_none());
    }
    #[test]
    fn registry_lifecycle() {
        #[derive(Clone, Debug, Default)]
        struct D {
            x: f32,
        }
        impl Reflect for D {
            fn type_name(&self) -> &'static str {
                "D"
            }
            fn fields(&self) -> Vec<(&'static str, String)> {
                vec![("x", format!("{:?}", self.x))]
            }
            fn clone_reflect(&self) -> Box<dyn Reflect> {
                Box::new(self.clone())
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }
        let mut r = TypeRegistry::new();
        r.register::<D>();
        assert!(r.get_by_name("D").is_some());
        assert_eq!(
            r.create_default("D")
                .unwrap()
                .as_any()
                .downcast_ref::<D>()
                .unwrap()
                .x,
            0.0
        );
    }
    #[test]
    fn type_info_prim() {
        let i = 42_f32.type_info();
        assert_eq!(i.name, "f32");
        assert!(i.fields.is_empty());
    }
}
