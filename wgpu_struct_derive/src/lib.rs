use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Data, DeriveInput, Error, Index, parse_macro_input};

#[proc_macro_derive(GpuLayout)]
pub fn derive_gpu_layout(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    derive_gpu_layout_impl(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn derive_gpu_layout_impl(input: DeriveInput) -> Result<TokenStream, Error> {
    let data = if let Data::Struct(data) = input.data {
        data
    } else {
        return Err(Error::new_spanned(
            input,
            "#[derive(GpuLayout)] may only be used on structs",
        ));
    };

    let ident = input.ident;

    let field_tys = data.fields.iter().map(|field| &field.ty);

    let alignment = field_tys.clone().fold(
        quote! { 1_usize },
        |acc, ty| quote! { wgpu_struct::__internal::max(#acc, < #ty as wgpu_struct::GpuLayout >::ALIGNMENT) },
    );

    Ok(quote! {
        impl wgpu_struct::GpuLayout for #ident {
            const ALIGNMENT: usize = #alignment;
            const SIZE: Option<usize> = Some(0_usize #( + < #field_tys as wgpu_struct::GpuLayout >::SIZE.unwrap() )*);
        }
        const _: Option<usize> = <#ident as wgpu_struct::GpuLayout>::SIZE;
    })
    .into()
}

#[proc_macro_derive(GpuEncode)]
pub fn derive_gpu_encode(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    derive_gpu_encode_impl(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn derive_gpu_encode_impl(input: DeriveInput) -> Result<TokenStream, Error> {
    let data = if let Data::Struct(data) = input.data {
        data
    } else {
        return Err(Error::new_spanned(
            input,
            "#[derive(GpuEncode)] may only be used on structs",
        ));
    };

    let ident = input.ident;

    let fields = data
        .fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let ty = &field.ty;
            let ident = field
                .ident
                .clone()
                .map(|ident| ident.into_token_stream())
                .unwrap_or_else(|| Index::from(i).into_token_stream());

            quote! {
                <#ty as wgpu_struct::GpuEncode>::encode(&self.#ident, encoder)?;
            }
        })
        .collect::<Vec<_>>();

    Ok(quote! {
        impl wgpu_struct::GpuEncode for #ident {
            fn encode(&self, encoder: &mut wgpu_struct::GpuEncoder<impl std::io::Write>) -> std::io::Result<()> {
                encoder.struct_align::<Self>(|encoder| {
                    #(#fields)*
                    Ok(())
                })
            }
        }
    })
    .into()
}

#[proc_macro_derive(GpuDecode)]
pub fn derive_gpu_decode(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    derive_gpu_decode_impl(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn derive_gpu_decode_impl(input: DeriveInput) -> Result<TokenStream, Error> {
    let data = if let Data::Struct(data) = input.data {
        data
    } else {
        return Err(Error::new_spanned(
            input,
            "#[derive(GpuDecode)] may only be used on structs",
        ));
    };

    let ident = input.ident;

    let body = match data.fields {
        syn::Fields::Named(fields) => {
            let inner = fields.named.iter().map(|field| {
                let name = field.ident.as_ref().unwrap();
                let ty = &field.ty;
                quote! {
                    #name: <#ty as wgpu_struct::GpuDecode>::decode(decoder)?,
                }
            });
            quote! { Self { #(#inner)* } }
        }
        syn::Fields::Unnamed(fields) => {
            let inner = fields.unnamed.iter().map(|field| {
                let ty = &field.ty;
                quote! {
                    <#ty as wgpu_struct::GpuDecode>::decode(decoder)?,
                }
            });
            quote! { Self (#(#inner)*) }
        }
        syn::Fields::Unit => quote! { Self },
    };

    Ok(quote! {
        impl wgpu_struct::GpuDecode for #ident {
            fn decode(decoder: &mut wgpu_struct::GpuDecoder<impl std::io::Read>) -> std::io::Result<Self> {
                decoder.struct_align::<Self>(|decoder| Ok(#body))
            }
        }
    })
    .into()
}
