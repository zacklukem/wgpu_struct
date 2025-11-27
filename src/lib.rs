use smallvec::{SmallVec, smallvec};
use std::{
    io::{Result, Write},
    mem,
};

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

    pub fn struct_scope(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let prev_align = mem::replace(&mut self.align, 1);
        f(self)?;
        let struct_align = mem::replace(&mut self.align, prev_align);
        self.align_to(struct_align)?;
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
                    encoder.align_to(Self::ALIGNMENT)?;
                    encoder.write_all(bytemuck::bytes_of(self))
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
                    encoder.align_to(Self::ALIGNMENT)?;
                    let ($($i,)*) = self;
                    $(
                        $i.encode(encoder)?;
                    )*
                    encoder.align_to(Self::ALIGNMENT)
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

pub fn gpu_encode_vec<T: GpuEncode>(value: &T) -> Vec<u8> {
    let mut encoder = GpuEncoder::new(Vec::new());
    value.encode(&mut encoder).unwrap();
    encoder.end().unwrap()
}
