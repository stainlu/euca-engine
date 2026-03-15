use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive macro for the `Reflect` trait.
///
/// Generates runtime type information including field names and string representations.
///
/// ```ignore
/// #[derive(Reflect)]
/// struct Health {
///     current: f32,
///     max: f32,
/// }
/// ```
#[proc_macro_derive(Reflect)]
pub fn derive_reflect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    let fields_impl = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_entries: Vec<_> = fields
                    .named
                    .iter()
                    .map(|f| {
                        let field_name = f.ident.as_ref().unwrap();
                        let field_name_str = field_name.to_string();
                        quote! {
                            (#field_name_str, format!("{:?}", self.#field_name))
                        }
                    })
                    .collect();
                quote! {
                    vec![#(#field_entries),*]
                }
            }
            _ => quote! { vec![] },
        },
        _ => quote! { vec![] },
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics euca_reflect::Reflect for #name #ty_generics #where_clause {
            fn type_name(&self) -> &'static str {
                #name_str
            }

            fn fields(&self) -> Vec<(&'static str, String)> {
                #fields_impl
            }
        }
    };

    TokenStream::from(expanded)
}
