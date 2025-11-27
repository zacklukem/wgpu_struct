use smallvec::{SmallVec, smallvec};
use std::io::{Result, Write};

pub trait GpuLayout {
    const ALIGNMENT: usize;
    const SIZE: Option<usize>;
}

pub trait GpuEncode: GpuLayout {
    fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()>;
}

pub struct GpuEncoder<W: Write> {
    buffer: W,
    written: usize,
    align: usize,
}

impl<W: Write> GpuEncoder<W> {
    fn new(buffer: W) -> Self {
        GpuEncoder {
            buffer,
            written: 0,
            align: 1,
        }
    }

    pub fn align_to(&mut self, align: usize) -> Result<()> {
        let padding = self.written.next_multiple_of(align) - self.written;

        if padding == 0 {
            return Ok(());
        }

        let padding: SmallVec<[u8; 16]> = smallvec![0; padding];
        self.write_all(&padding)?;
        self.align = self.align.max(align);
        Ok(())
    }

    pub fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        self.buffer.write_all(buf)?;
        self.written += buf.len();
        Ok(())
    }

    pub fn aligned_to<T: GpuEncode>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        self.align_to(T::ALIGNMENT)?;
        f(self)?;
        self.align_to(T::ALIGNMENT)?;
        Ok(())
    }

    fn end(mut self) -> Result<W> {
        self.align_to(self.align)?;
        Ok(self.buffer)
    }
}

macro_rules! impl_simple_primitives {
    ($($type:ty, $align:expr, $size:expr;)*) => {
        $(
            impl GpuLayout for $type {
                const ALIGNMENT: usize = $align;
                const SIZE: Option<usize> = Some($size);
            }

            impl GpuEncode for $type {
                fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()> {
                    encoder.aligned_to::<Self>(|encoder| encoder.write_all(bytemuck::bytes_of(self)))
                }
            }
        )*
    };
}

impl_simple_primitives! {
    f32, 4, 4;
    i32, 4, 4;
    u32, 4, 4;
}

macro_rules! impl_vectors {
    ($($type:ty, ($($i:ident),*), $align:expr, $size:expr;)*) => {
        $(
            impl GpuLayout for $type {
                const ALIGNMENT: usize = $align;
                const SIZE: Option<usize> = Some($size);
            }

            impl GpuEncode for $type {
                fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()> {
                    encoder.aligned_to::<Self>(|encoder| {
                        let ($($i,)*) = self;
                        $(
                            $i.encode(encoder)?;
                        )*
                        Ok(())
                    })
                }
            }
        )*
    };
}

impl_vectors! {
    // vec*f
    (f32, f32), (a, b), 8, 8;
    (f32, f32, f32), (a, b, c), 16, 12;
    (f32, f32, f32, f32), (a, b, c, d), 16, 16;

    // vec*u
    (u32, u32), (a, b), 8, 8;
    (u32, u32, u32), (a, b, c), 16, 12;
    (u32, u32, u32, u32), (a, b, c, d), 16, 16;

    // mat*f
    ((f32, f32), (f32, f32)), (a, b), 8, 16; // mat2x2

    ((f32, f32), (f32, f32), (f32, f32)), (a, b, c), 8, 24; // mat3x2
    ((f32, f32, f32), (f32, f32, f32)), (a, b), 16, 32; // mat2x3
    ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)), (a, b, c), 16, 48; // mat3x3

    ((f32, f32), (f32, f32), (f32, f32), (f32, f32)), (a, b, c, d), 8, 32; // mat4x2
    ((f32, f32, f32, f32), (f32, f32, f32, f32)), (a, b), 16, 32; // mat2x4
    ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32), (f32, f32, f32)), (a, b, c, d), 16, 64; // mat4x3
    ((f32, f32, f32, f32), (f32, f32, f32, f32), (f32, f32, f32, f32)), (a, b, c), 16, 48; // mat3x4
    ((f32, f32, f32, f32), (f32, f32, f32, f32), (f32, f32, f32, f32), (f32, f32, f32, f32)), (a, b, c, d), 16, 64; // mat4x4

    // mat*u
    ((u32, u32), (u32, u32)), (a, b), 8, 16; // mat2x2

    ((u32, u32), (u32, u32), (u32, u32)), (a, b, c), 8, 24; // mat3x2
    ((u32, u32, u32), (u32, u32, u32)), (a, b), 16, 32; // mat2x3
    ((u32, u32, u32), (u32, u32, u32), (u32, u32, u32)), (a, b, c), 16, 48; // mat3x3

    ((u32, u32), (u32, u32), (u32, u32), (u32, u32)), (a, b, c, d), 8, 32; // mat4x2
    ((u32, u32, u32, u32), (u32, u32, u32, u32)), (a, b), 16, 32; // mat2x4
    ((u32, u32, u32), (u32, u32, u32), (u32, u32, u32), (u32, u32, u32)), (a, b, c, d), 16, 64; // mat4x3
    ((u32, u32, u32, u32), (u32, u32, u32, u32), (u32, u32, u32, u32)), (a, b, c), 16, 48; // mat3x4
    ((u32, u32, u32, u32), (u32, u32, u32, u32), (u32, u32, u32, u32), (u32, u32, u32, u32)), (a, b, c, d), 16, 64; // mat4x4
}

impl<T: GpuLayout> GpuLayout for Vec<T> {
    const ALIGNMENT: usize = T::ALIGNMENT;
    const SIZE: Option<usize> = None;
}

impl<T: GpuEncode> GpuEncode for Vec<T> {
    fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()> {
        for item in self {
            item.encode(encoder)?;
        }
        Ok(())
    }
}

impl<T: GpuLayout> GpuLayout for &[T] {
    const ALIGNMENT: usize = T::ALIGNMENT;
    const SIZE: Option<usize> = None;
}

impl<T: GpuEncode> GpuEncode for &[T] {
    fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()> {
        for item in *self {
            item.encode(encoder)?;
        }
        Ok(())
    }
}

impl<T: GpuLayout, const N: usize> GpuLayout for [T; N] {
    const ALIGNMENT: usize = T::ALIGNMENT;
    const SIZE: Option<usize> = if let Some(size) = T::SIZE {
        Some(size.next_multiple_of(T::ALIGNMENT) * N)
    } else {
        None
    };
}

impl<T: GpuEncode, const N: usize> GpuEncode for [T; N] {
    fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()> {
        for item in self {
            item.encode(encoder)?;
        }
        Ok(())
    }
}

pub fn gpu_encode<T: GpuEncode, W: Write>(container: W, value: &T) -> Result<W> {
    let mut encoder = GpuEncoder::new(container);
    value.encode(&mut encoder)?;
    encoder.end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_vec2() {
        let encoded = gpu_encode(vec![], &(0.5, 0.2)).unwrap();
        assert_eq!(&encoded, bytemuck::bytes_of(&[0.5_f32, 0.2]));
        assert_eq!(encoded.len(), 8);
    }

    #[test]
    fn encodes_vec3() {
        let encoded = gpu_encode(vec![], &(0.5, 0.2, 0.3)).unwrap();
        assert_eq!(&encoded, bytemuck::bytes_of(&[0.5_f32, 0.2, 0.3, 0.0]));
        assert_eq!(encoded.len(), 16);
    }

    #[test]
    fn encodes_vec4() {
        let encoded = gpu_encode(vec![], &(0.5, 0.2, 0.3, 0.8)).unwrap();
        assert_eq!(&encoded, bytemuck::bytes_of(&[0.5_f32, 0.2, 0.3, 0.8]));
        assert_eq!(encoded.len(), 16);
    }
}
