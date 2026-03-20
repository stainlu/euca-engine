use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(Reflect)]
pub fn derive_reflect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let ns = name.to_string();
    let (ig, tg, wc) = input.generics.split_for_impl();
    let expanded = match &input.data {
        Data::Struct(d) => match &d.fields {
            Fields::Named(f) => gen_named(name, &ns, &ig, &tg, wc, f),
            Fields::Unnamed(f) => gen_tuple(name, &ns, &ig, &tg, wc, f),
            Fields::Unit => gen_leaf(name, &ns, &ig, &tg, wc),
        },
        Data::Enum(d) => gen_enum(name, &ns, &ig, &tg, wc, d),
        Data::Union(_) => return syn::Error::new_spanned(name, "Reflect cannot be derived for unions").to_compile_error().into(),
    };
    TokenStream::from(expanded)
}
fn cm() -> proc_macro2::TokenStream {
    quote! { fn clone_reflect(&self) -> Box<dyn euca_reflect::Reflect> { Box::new(self.clone()) } fn as_any(&self) -> &dyn std::any::Any { self } fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self } }
}
fn gen_named(name: &syn::Ident, ns: &str, ig: &syn::ImplGenerics<'_>, tg: &syn::TypeGenerics<'_>, wc: Option<&syn::WhereClause>, fields: &syn::FieldsNamed) -> proc_macro2::TokenStream {
    let entries: Vec<_> = fields.named.iter().map(|f| { let i = f.ident.as_ref().unwrap(); let s = i.to_string(); quote!{(#s, format!("{:?}", self.#i))} }).collect();
    let ra: Vec<_> = fields.named.iter().map(|f| { let i = f.ident.as_ref().unwrap(); let s = i.to_string(); quote!{#s => Some(&self.#i as &dyn euca_reflect::Reflect)} }).collect();
    let ma: Vec<_> = fields.named.iter().map(|f| { let i = f.ident.as_ref().unwrap(); let s = i.to_string(); quote!{#s => Some(&mut self.#i as &mut dyn euca_reflect::Reflect)} }).collect();
    let sa: Vec<_> = fields.named.iter().map(|f| { let i = f.ident.as_ref().unwrap(); let s = i.to_string(); let t = &f.ty; quote!{#s => { if let Some(v) = value.as_any().downcast_ref::<#t>() { self.#i = v.clone(); true } else { false } }} }).collect();
    let fi: Vec<_> = fields.named.iter().map(|f| { let i = f.ident.as_ref().unwrap(); let s = i.to_string(); quote!{euca_reflect::FieldInfo{name:#s,type_name:self.#i.type_name()}} }).collect();
    let c = cm();
    quote! { impl #ig euca_reflect::Reflect for #name #tg #wc {
        fn type_name(&self) -> &'static str { #ns }
        fn fields(&self) -> Vec<(&'static str, String)> { vec![#(#entries),*] }
        fn field_ref(&self, name: &str) -> Option<&dyn euca_reflect::Reflect> { match name { #(#ra,)* _ => None } }
        fn field_mut(&mut self, name: &str) -> Option<&mut dyn euca_reflect::Reflect> { match name { #(#ma,)* _ => None } }
        fn set_field(&mut self, name: &str, value: &dyn euca_reflect::Reflect) -> bool { match name { #(#sa)* _ => false } }
        fn type_info(&self) -> euca_reflect::TypeInfo { euca_reflect::TypeInfo{name:#ns,fields:vec![#(#fi),*]} }
        #c
    }}
}
fn gen_tuple(name: &syn::Ident, ns: &str, ig: &syn::ImplGenerics<'_>, tg: &syn::TypeGenerics<'_>, wc: Option<&syn::WhereClause>, fields: &syn::FieldsUnnamed) -> proc_macro2::TokenStream {
    let n = fields.unnamed.len();
    let entries: Vec<_> = (0..n).map(|i| { let idx = syn::Index::from(i); let s = i.to_string(); quote!{(#s, format!("{:?}", self.#idx))} }).collect();
    let ra: Vec<_> = (0..n).map(|i| { let idx = syn::Index::from(i); let s = i.to_string(); quote!{#s => Some(&self.#idx as &dyn euca_reflect::Reflect)} }).collect();
    let ma: Vec<_> = (0..n).map(|i| { let idx = syn::Index::from(i); let s = i.to_string(); quote!{#s => Some(&mut self.#idx as &mut dyn euca_reflect::Reflect)} }).collect();
    let sa: Vec<_> = fields.unnamed.iter().enumerate().map(|(i,f)| { let idx = syn::Index::from(i); let s = i.to_string(); let t = &f.ty; quote!{#s => { if let Some(v) = value.as_any().downcast_ref::<#t>() { self.#idx = v.clone(); true } else { false } }} }).collect();
    let fi: Vec<_> = (0..n).map(|i| { let idx = syn::Index::from(i); let s = i.to_string(); quote!{euca_reflect::FieldInfo{name:#s,type_name:self.#idx.type_name()}} }).collect();
    let c = cm();
    quote! { impl #ig euca_reflect::Reflect for #name #tg #wc {
        fn type_name(&self) -> &'static str { #ns }
        fn fields(&self) -> Vec<(&'static str, String)> { vec![#(#entries),*] }
        fn field_ref(&self, name: &str) -> Option<&dyn euca_reflect::Reflect> { match name { #(#ra,)* _ => None } }
        fn field_mut(&mut self, name: &str) -> Option<&mut dyn euca_reflect::Reflect> { match name { #(#ma,)* _ => None } }
        fn set_field(&mut self, name: &str, value: &dyn euca_reflect::Reflect) -> bool { match name { #(#sa)* _ => false } }
        fn type_info(&self) -> euca_reflect::TypeInfo { euca_reflect::TypeInfo{name:#ns,fields:vec![#(#fi),*]} }
        #c
    }}
}
fn gen_leaf(name: &syn::Ident, ns: &str, ig: &syn::ImplGenerics<'_>, tg: &syn::TypeGenerics<'_>, wc: Option<&syn::WhereClause>) -> proc_macro2::TokenStream {
    let c = cm();
    quote! { impl #ig euca_reflect::Reflect for #name #tg #wc { fn type_name(&self) -> &'static str { #ns } fn fields(&self) -> Vec<(&'static str, String)> { Vec::new() } #c } }
}
fn gen_enum(name: &syn::Ident, ns: &str, ig: &syn::ImplGenerics<'_>, tg: &syn::TypeGenerics<'_>, wc: Option<&syn::WhereClause>, data: &syn::DataEnum) -> proc_macro2::TokenStream {
    let arms: Vec<_> = data.variants.iter().map(|v| { let vi = &v.ident; let vs = vi.to_string(); match &v.fields { Fields::Unit => quote!{#name::#vi => #vs}, Fields::Unnamed(_) => quote!{#name::#vi(..) => #vs}, Fields::Named(_) => quote!{#name::#vi{..} => #vs} } }).collect();
    let c = cm();
    quote! { impl #ig euca_reflect::Reflect for #name #tg #wc { fn type_name(&self) -> &'static str { #ns } fn fields(&self) -> Vec<(&'static str, String)> { vec![("variant", match self { #(#arms,)* }.to_string())] } #c } }
}
