use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    parse::Parser,
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
    Attribute, Error, Expr, FnArg, ImplItem, ImplItemFn, ItemFn, ItemImpl, Lit, Meta, Pat,
    ReturnType, Type, TypePath,
};

#[proc_macro_attribute]
pub fn kernel_function(attr: TokenStream, item: TokenStream) -> TokenStream {
    let parser = Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    let metas = parser.parse(attr.into()).unwrap_or_else(|_| Punctuated::new());
    let mut function = parse_macro_input!(item as ItemFn);

    match expand_kernel_function(metas.into_iter().collect(), &mut function) {
        Ok(expansion) => (
            quote! {
                #function
                #expansion
            }
        )
        .into(),
        Err(error) => error.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn kernel_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        let error = Error::new(Span::call_site(), "kernel_module does not take arguments");
        return error.to_compile_error().into();
    }

    let mut item_impl = parse_macro_input!(item as ItemImpl);
    match expand_kernel_module(&mut item_impl) {
        Ok(expansion) => (
            quote! {
                #item_impl
                #expansion
            }
        )
        .into(),
        Err(error) => error.to_compile_error().into(),
    }
}

struct KernelMeta {
    kernel_name: String,
    description: Option<String>,
}

type MetaList = Vec<Meta>;

fn parse_kernel_meta(
    args: MetaList,
    attrs: &mut Vec<Attribute>,
    fallback: &Ident,
) -> Result<KernelMeta, Error> {
    let mut kernel_name: Option<String> = None;
    let mut description: Option<String> = None;

    for meta in args {
        match meta {
            Meta::NameValue(kv) if kv.path.is_ident("name") => {
                kernel_name = Some(expect_string_literal(&kv.value)?);
            }
            Meta::NameValue(kv) if kv.path.is_ident("description") => {
                description = Some(expect_string_literal(&kv.value)?);
            }
            other => return Err(Error::new_spanned(other, "unsupported attribute argument")),
        }
    }

    attrs.retain(|attr| {
        if attr.path().is_ident("description") {
            if let Ok(value) = attr.parse_args::<syn::LitStr>() {
                description = Some(value.value());
                false
            } else {
                true
            }
        } else {
            true
        }
    });

    Ok(KernelMeta {
        kernel_name: kernel_name.unwrap_or_else(|| fallback.to_string()),
        description,
    })
}

fn expect_string_literal(expr: &Expr) -> Result<String, Error> {
    if let Expr::Lit(expr_lit) = expr {
        if let Lit::Str(lit_str) = &expr_lit.lit {
            return Ok(lit_str.value());
        }
    }

    Err(Error::new(expr.span(), "expected string literal"))
}

struct ParameterMeta {
    ident: Ident,
    ty: Type,
    schema_ty: Type,
    description: Option<String>,
    default: Option<Expr>,
    optional: bool,
}

fn parse_parameters(inputs: &mut Punctuated<FnArg, syn::Token![,]>) -> Result<Vec<ParameterMeta>, Error> {
    let mut params = Vec::new();

    for arg in inputs.iter_mut() {
        let FnArg::Typed(pat_ty) = arg else { continue };

        let Pat::Ident(pat_ident) = &*pat_ty.pat else {
            return Err(Error::new_spanned(&pat_ty.pat, "unsupported parameter pattern"));
        };

        let mut description = None;
        let mut default = None;
        let mut optional = false;
        let mut retained_attrs = Vec::new();

        for attr in &pat_ty.attrs {
            if attr.path().is_ident("description") {
                if let Ok(value) = attr.parse_args::<syn::LitStr>() {
                    description = Some(value.value());
                    continue;
                }
            }

            if attr.path().is_ident("param") {
                let parsed = attr.parse_args_with(
                    Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                )?;
                for entry in parsed {
                    match entry {
                        Meta::NameValue(kv) if kv.path.is_ident("description") => {
                            description = Some(expect_string_literal(&kv.value)?);
                        }
                        Meta::NameValue(kv) if kv.path.is_ident("default") => {
                            default = Some(kv.value.clone());
                        }
                        Meta::Path(path) if path.is_ident("optional") => {
                            optional = true;
                        }
                        other => {
                            return Err(Error::new_spanned(other, "unsupported parameter attribute"));
                        }
                    }
                }
                continue;
            }

            retained_attrs.push(attr.clone());
        }

        let (schema_ty, is_option) = extract_schema_type(&pat_ty.ty);
        if is_option {
            optional = true;
        }

        pat_ty.attrs = retained_attrs;

        params.push(ParameterMeta {
            ident: pat_ident.ident.clone(),
            ty: (*pat_ty.ty).clone(),
            schema_ty,
            description,
            default,
            optional,
        });
    }

    Ok(params)
}

fn extract_schema_type(ty: &Type) -> (Type, bool) {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(segment) = path.segments.last() {
            if segment.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(angle) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = angle.args.first() {
                        return (inner.clone(), true);
                    }
                }
            }
        }
    }

    (ty.clone(), false)
}

#[derive(Clone)]
enum ReturnKind {
    Unit,
    Value,
    Result,
}

fn return_kind(output: &ReturnType) -> ReturnKind {
    match output {
        ReturnType::Default => ReturnKind::Unit,
        ReturnType::Type(_, ty) => {
            if parse_result_type(ty).is_some() {
                ReturnKind::Result
            } else {
                ReturnKind::Value
            }
        }
    }
}

fn parse_result_type(ty: &Type) -> Option<(Type, Type)> {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(segment) = path.segments.last() {
            if segment.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    let mut iter = args.args.iter();
                    let ok = iter.next()?;
                    let err = iter.next()?;
                    if let (syn::GenericArgument::Type(ok_ty), syn::GenericArgument::Type(err_ty)) = (ok, err) {
                        return Some((ok_ty.clone(), err_ty.clone()));
                    }
                }
            }
        }
    }
    None
}

fn expand_kernel_function(args: MetaList, function: &mut ItemFn) -> Result<TokenStream2, Error> {
    let original_ident = function.sig.ident.clone();
    let KernelMeta {
        kernel_name,
        description,
    } = parse_kernel_meta(args, &mut function.attrs, &original_ident)?;

    let params = parse_parameters(&mut function.sig.inputs)?;

    let args_struct_ident = format_ident!(
        "__{}_Args",
        original_ident.to_string().to_uppercase()
    );
    let wrapper_ident = format_ident!("{}KernelFunction", original_ident.to_string().to_uppercase());
    let definition_ident = format_ident!("__{}_definition", original_ident.to_string().to_uppercase());
    let export_ident = format_ident!("{}_kernel", original_ident);

    let mut args_fields = Vec::new();
    let mut assignment_idents = Vec::new();
    let mut helper_items = Vec::new();
    let mut definition_statements = Vec::new();

    for param in &params {
        let ident = &param.ident;
        assignment_idents.push(ident.clone());

        let field_ty = &param.ty;
        let schema_ty = &param.schema_ty;

        let description_expr = param
            .description
            .as_ref()
            .map(|value| quote! { .with_description(#value) })
            .unwrap_or_else(TokenStream2::new);

        let mut parameter_expr = quote! {
            ::denkwerk::functions::FunctionParameter::new(stringify!(#ident), ::denkwerk::functions::json_schema_for::<#schema_ty>())
                #description_expr
        };

        let mut field_attrs = Vec::new();

        if param.optional || param.default.is_some() {
            parameter_expr = quote! { #parameter_expr.optional() };
        }

        if let Some(default_expr) = &param.default {
            let helper_ident = format_ident!(
                "__{}_{}_default",
                original_ident.to_string().to_uppercase(),
                ident.to_string().to_uppercase()
            );
            helper_items.push(quote! {
                fn #helper_ident() -> #field_ty {
                    #default_expr
                }
            });
            field_attrs.push(quote! { #[serde(default = #helper_ident)] });
            parameter_expr = quote! {
                #parameter_expr.with_default(::denkwerk::functions::to_value(#helper_ident()))
            };
        } else if param.optional {
            field_attrs.push(quote! { #[serde(default)] });
        }

        args_fields.push(quote! {
            #(#field_attrs)*
            pub #ident: #field_ty,
        });

        definition_statements.push(quote! {
            definition.add_parameter(#parameter_expr);
        });
    }

    let description_statement = description
        .as_ref()
        .map(|text| quote! { definition = definition.with_description(#text); })
        .unwrap_or_else(TokenStream2::new);

    let args_struct = quote! {
        #[derive(::serde::Deserialize)]
        struct #args_struct_ident {
            #(#args_fields)*
        }
    };

    helper_items.push(quote! {
        fn #definition_ident() -> ::denkwerk::FunctionDefinition {
            let mut definition = ::denkwerk::FunctionDefinition::new(#kernel_name);
            #(
                #definition_statements
            )*
            #description_statement
            definition
        }
    });

    let return_kind = return_kind(&function.sig.output);
    let call_expr = if function.sig.asyncness.is_some() {
        quote! { #original_ident(#(#assignment_idents),*).await }
    } else {
        quote! { #original_ident(#(#assignment_idents),*) }
    };

    let invoke_body = build_invoke_body(&return_kind, call_expr, &kernel_name);

    let expansion = quote! {
        #args_struct
        #(#helper_items)*

        pub struct #wrapper_ident;

        #[::async_trait::async_trait]
        impl ::denkwerk::functions::KernelFunction for #wrapper_ident {
            fn definition(&self) -> ::denkwerk::FunctionDefinition {
                #definition_ident()
            }

            async fn invoke(&self, arguments: ::serde_json::Value) -> Result<::serde_json::Value, ::denkwerk::LLMError> {
                let args: #args_struct_ident = ::serde_json::from_value(arguments)
                    .map_err(|error| ::denkwerk::LLMError::InvalidFunctionArguments(error.to_string()))?;
                let #args_struct_ident { #(#assignment_idents),* } = args;
                #invoke_body
            }
        }

        pub fn #export_ident() -> ::denkwerk::DynKernelFunction {
            ::std::sync::Arc::new(#wrapper_ident)
        }
    };

    Ok(expansion)
}

fn build_invoke_body(return_kind: &ReturnKind, call: TokenStream2, kernel_name: &str) -> TokenStream2 {
    match return_kind {
        ReturnKind::Unit => {
            quote! {
                #call;
                Ok(::serde_json::Value::Null)
            }
        }
        ReturnKind::Value => {
            quote! {
                let value = #call;
                ::serde_json::to_value(value).map_err(::denkwerk::LLMError::from)
            }
        }
        ReturnKind::Result => {
            quote! {
                match #call {
                    Ok(value) => ::serde_json::to_value(value).map_err(::denkwerk::LLMError::from),
                    Err(error) => Err(::denkwerk::LLMError::FunctionExecution {
                        function: #kernel_name.to_string(),
                        message: error.to_string(),
                    }),
                }
            }
        }
    }
}

fn expand_kernel_module(item_impl: &mut ItemImpl) -> Result<TokenStream2, Error> {
    let mut expansions = Vec::new();
    let mut register_statements = Vec::new();
    let self_ty = &*item_impl.self_ty;

    for item in item_impl.items.iter_mut() {
        let ImplItem::Fn(method) = item else { continue };

        let mut kernel_attr_index = None;
        for (index, attr) in method.attrs.iter().enumerate() {
            if attr.path().is_ident("kernel_function") {
                kernel_attr_index = Some(index);
                break;
            }
        }

        let Some(index) = kernel_attr_index else { continue };

        let attr = method.attrs.remove(index);
        let parsed = attr.parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)?;
        let args: MetaList = parsed.into_iter().collect();

        let expansion = expand_kernel_method(args, method, self_ty)?;
        expansions.push(expansion.tokens);
        register_statements.push(expansion.register_stmt);
    }

    if expansions.is_empty() {
        return Ok(TokenStream2::new());
    }

    let register_impl = quote! {
        impl #self_ty {
            pub fn kernel_functions(self: ::std::sync::Arc<Self>) -> Vec<::denkwerk::DynKernelFunction> {
                let mut functions = Vec::new();
                #(#register_statements)*
                functions
            }

            pub fn register_kernel_functions(self: ::std::sync::Arc<Self>, registry: &mut ::denkwerk::FunctionRegistry) {
                for function in self.kernel_functions() {
                    registry.register(function);
                }
            }
        }
    };

    Ok(quote! {
        #(#expansions)*
        #register_impl
    })
}

struct MethodExpansion {
    tokens: TokenStream2,
    register_stmt: TokenStream2,
}

fn expand_kernel_method(
    args: MetaList,
    method: &mut ImplItemFn,
    self_ty: &Type,
) -> Result<MethodExpansion, Error> {
    let method_ident = method.sig.ident.clone();
    let KernelMeta {
        kernel_name,
        description,
    } = parse_kernel_meta(args, &mut method.attrs, &method_ident)?;

    let has_self = method
        .sig
        .inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    if !has_self {
        return Err(Error::new(method.sig.span(), "kernel methods must take self"));
    }

    let params = parse_parameters(&mut method.sig.inputs)?;

    let args_struct_ident = format_ident!(
        "__{}_{}_Args",
        type_token(self_ty),
        method_ident.to_string().to_uppercase()
    );
    let wrapper_ident = format_ident!(
        "{}{}KernelFunction",
        type_token(self_ty),
        method_ident.to_string().to_uppercase()
    );
    let definition_ident = format_ident!(
        "__{}_{}_definition",
        type_token(self_ty),
        method_ident.to_string().to_uppercase()
    );

    let mut args_fields = Vec::new();
    let mut assignment_idents = Vec::new();
    let mut helper_items = Vec::new();
    let mut definition_statements = Vec::new();

    for param in &params {
        let ident = &param.ident;
        assignment_idents.push(ident.clone());
        let field_ty = &param.ty;
        let schema_ty = &param.schema_ty;
        let description_expr = param
            .description
            .as_ref()
            .map(|text| quote! { .with_description(#text) })
            .unwrap_or_else(TokenStream2::new);

        let mut parameter_expr = quote! {
            ::denkwerk::functions::FunctionParameter::new(stringify!(#ident), ::denkwerk::functions::json_schema_for::<#schema_ty>())
                #description_expr
        };

        let mut field_attrs = Vec::new();

        if param.optional || param.default.is_some() {
            parameter_expr = quote! { #parameter_expr.optional() };
        }

        if let Some(default_expr) = &param.default {
            let helper_ident = format_ident!(
                "__{}_{}_{}_default",
                type_token(self_ty),
                method_ident.to_string().to_uppercase(),
                ident.to_string().to_uppercase()
            );
            helper_items.push(quote! {
                fn #helper_ident() -> #field_ty {
                    #default_expr
                }
            });
            field_attrs.push(quote! { #[serde(default = #helper_ident)] });
            parameter_expr = quote! {
                #parameter_expr.with_default(::denkwerk::functions::to_value(#helper_ident()))
            };
        } else if param.optional {
            field_attrs.push(quote! { #[serde(default)] });
        }

        args_fields.push(quote! {
            #(#field_attrs)*
            pub #ident: #field_ty,
        });

        definition_statements.push(quote! {
            definition.add_parameter(#parameter_expr);
        });
    }

    let description_statement = description
        .as_ref()
        .map(|text| quote! { definition = definition.with_description(#text); })
        .unwrap_or_else(TokenStream2::new);

    let args_struct = quote! {
        #[derive(::serde::Deserialize)]
        struct #args_struct_ident {
            #(#args_fields)*
        }
    };

    helper_items.push(quote! {
        fn #definition_ident() -> ::denkwerk::FunctionDefinition {
            let mut definition = ::denkwerk::FunctionDefinition::new(#kernel_name);
            #(
                #definition_statements
            )*
            #description_statement
            definition
        }
    });

    let return_kind = return_kind(&method.sig.output);
    let call_expr = if method.sig.asyncness.is_some() {
        quote! { self.instance.#method_ident(#(#assignment_idents),*).await }
    } else {
        quote! { self.instance.#method_ident(#(#assignment_idents),*) }
    };

    let invoke_body = build_invoke_body(&return_kind, call_expr, &kernel_name);

    let tokens = quote! {
        #args_struct
        #(#helper_items)*

        struct #wrapper_ident {
            instance: ::std::sync::Arc<#self_ty>,
        }

        #[::async_trait::async_trait]
        impl ::denkwerk::functions::KernelFunction for #wrapper_ident {
            fn definition(&self) -> ::denkwerk::FunctionDefinition {
                #definition_ident()
            }

            async fn invoke(&self, arguments: ::serde_json::Value) -> Result<::serde_json::Value, ::denkwerk::LLMError> {
                let args: #args_struct_ident = ::serde_json::from_value(arguments)
                    .map_err(|error| ::denkwerk::LLMError::InvalidFunctionArguments(error.to_string()))?;
                let #args_struct_ident { #(#assignment_idents),* } = args;
                #invoke_body
            }
        }
    };

    let register_stmt = quote! {
        {
            let function: ::denkwerk::DynKernelFunction = ::std::sync::Arc::new(#wrapper_ident { instance: ::std::sync::Arc::clone(&self) });
            functions.push(function);
        }
    };

    Ok(MethodExpansion { tokens, register_stmt })
}

fn type_token(ty: &Type) -> String {
    let mut text = quote! { #ty }.to_string();
    text.retain(|c| c.is_alphanumeric() || c == '_');
    text
}
