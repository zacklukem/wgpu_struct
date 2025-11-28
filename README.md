# wgpu_struct

A wgsl data encoding and decoding library.

Because the layout of data types in wgsl differs from c or rust representation,
explicit handling of converting between host and gpu types is required. This
crate provides that functionality by serializing and deserializing data coming
to and from the gpu.

## Supported data types

This crate supports the following wgsl datatypes:
 * `u32`, `i32`, `f32`
   > Matches rust values
 * `vecN<f32>`, `vecN<i32>`, `vecN<u32>`
   > Equivilant to rust tuples
 * `matCxR<f32>`, `matCxR<i32>`, `matCxR<u32>`
   > Equivilant to rust nested tuples
 * `array<E, N>`
   > Equivilant to `[E; N]`
 * `array<E>`
   > Equivilant to `Vec<E>` (Or `&[E]` for encoding only)
 * Custom structs
   > Using [`GpuEncode`] and [`GpuDecode`] derive macros

## Examples

```rust
#[derive(GpuLayout, GpuEncode)]
struct Sphere {
    radius: f32,
    origin: (f32, f32, f32),
}

let data = vec![
    Sphere {
        radius: 1.0,
        origin: (1.0, 2.0, 3.0),
    },
    Sphere {
        radius: 2.0,
        origin: (4.0, 5.0, 6.0),
    },
];

let buffer = gpu_encode(Vec::new(), &data)?;

// [...] handle the buffer with the gpu here

let result = gpu_decode::<Vec<(u32, u32, u32)>>(Cursor::new(&gpu_output))?;
assert_eq!(result, vec![(1, 2, 3), (4, 5, 6)]);

```
