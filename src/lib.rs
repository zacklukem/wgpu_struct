use arrayvec::ArrayVec;
use smallvec::{SmallVec, smallvec};
use std::io::{ErrorKind, Read, Result, Write};

#[cfg(feature = "wgpu_struct_derive")]
pub use wgpu_struct_derive::{GpuEncode, GpuLayout};

#[cfg(feature = "wgpu_struct_derive")]
pub mod __internal {
    pub const fn max(a: usize, b: usize) -> usize {
        if a < b { b } else { a }
    }
}

pub trait GpuLayout {
    const ALIGNMENT: usize;
    const SIZE: Option<usize>;
}

pub trait GpuEncode: GpuLayout {
    fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()>;
}

pub trait GpuDecode: GpuLayout
where
    Self: Sized,
{
    fn decode(decoder: &mut GpuDecoder<impl Read>) -> Result<Self>;
}

pub struct GpuEncoder<W: Write> {
    buffer: W,
    written: usize,
    align: usize,
}

pub struct GpuDecoder<R: Read> {
    buffer: R,
    read: usize,
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
        self.align = self.align.max(align);
        let padding = self.written.next_multiple_of(align) - self.written;

        if padding == 0 {
            return Ok(());
        }

        let padding: SmallVec<[u8; 16]> = smallvec![0; padding];
        self.write_all(&padding)
    }

    pub fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        self.buffer.write_all(buf)?;
        self.written += buf.len();
        Ok(())
    }

    pub fn struct_align<T: GpuEncode>(
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

impl<R: Read> GpuDecoder<R> {
    fn new(buffer: R) -> Self {
        GpuDecoder { buffer, read: 0 }
    }

    pub fn aligned_to(&mut self, align: usize) -> Result<()> {
        let padding = self.read.next_multiple_of(align) - self.read;

        if padding == 0 {
            return Ok(());
        }

        let mut padding: SmallVec<[u8; 16]> = smallvec![0; padding];
        self.read_all(&mut padding)
    }

    pub fn read_all(&mut self, buf: &mut [u8]) -> Result<()> {
        self.buffer.read_exact(buf)?;
        self.read += buf.len();
        Ok(())
    }

    pub fn struct_align<T: GpuEncode>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.aligned_to(T::ALIGNMENT)?;
        let r = f(self)?;
        self.aligned_to(T::ALIGNMENT)?;
        Ok(r)
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
                    encoder.write_all(&self.to_le_bytes())
                }
            }

            impl GpuDecode for $type {
                fn decode(decoder: &mut GpuDecoder<impl Read>) -> Result<Self> {
                    decoder.aligned_to(Self::ALIGNMENT)?;
                    let mut buf = [0_u8; $size];
                    decoder.read_all(&mut buf)?;
                    Ok(Self::from_le_bytes(buf))
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
    ($(($($types:ty),*), ($($i:ident),*), $align:expr, $size:expr;)*) => {
        $(
            impl GpuLayout for ($($types,)*) {
                const ALIGNMENT: usize = $align;
                const SIZE: Option<usize> = Some($size);
            }

            impl GpuEncode for ($($types,)*) {
                fn encode(&self, encoder: &mut GpuEncoder<impl Write>) -> Result<()> {
                    encoder.align_to(Self::ALIGNMENT)?;
                    let ($($i,)*) = self;
                    $(
                        $i.encode(encoder)?;
                    )*
                    Ok(())
                }
            }

            impl GpuDecode for ($($types,)*) {
                fn decode(decoder: &mut GpuDecoder<impl Read>) -> Result<Self> {
                    decoder.aligned_to(Self::ALIGNMENT)?;
                    Ok(($(
                        <$types as GpuDecode>::decode(decoder)?,
                    )*))
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
        encoder.align_to(Self::ALIGNMENT)
    }
}

impl<T: GpuDecode> GpuDecode for Vec<T> {
    fn decode(decoder: &mut GpuDecoder<impl Read>) -> Result<Self> {
        let mut items = Vec::new();
        loop {
            let v = match T::decode(decoder) {
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
                Ok(v) => v,
            };
            items.push(v);
        }
        Ok(items)
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
        encoder.align_to(Self::ALIGNMENT)
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
        encoder.align_to(Self::ALIGNMENT)
    }
}

impl<T: GpuDecode, const N: usize> GpuDecode for [T; N] {
    fn decode(decoder: &mut GpuDecoder<impl Read>) -> Result<Self> {
        let mut items = ArrayVec::<T, N>::new();
        for _ in 0..N {
            items.push(T::decode(decoder)?);
        }
        Ok(items.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

pub fn gpu_decode<T: GpuDecode>(container: impl Read) -> Result<T> {
    let mut decoder = GpuDecoder::new(container);
    T::decode(&mut decoder)
}

pub fn gpu_encode<T: GpuEncode, W: Write>(container: W, value: &T) -> Result<W> {
    let mut encoder = GpuEncoder::new(container);
    value.encode(&mut encoder)?;
    encoder.end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    mod wgpu_struct {
        pub use crate::*;
    }

    macro_rules! words {
        ($ty: ty, $($v:expr),*) => {
            Vec::from([$(<$ty>::to_le_bytes($v),)*].as_flattened())
        };
    }

    #[test]
    fn encodes_u32() {
        let encoded = gpu_encode(vec![], &0xabcd1234_u32).unwrap();
        assert_eq!(&encoded, &[0x34, 0x12, 0xcd, 0xab]);
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn encodes_vec2() {
        let encoded = gpu_encode(vec![], &(0.5, 0.2)).unwrap();
        assert_eq!(encoded, words!(f32, 0.5, 0.2));
        assert_eq!(encoded.len(), 8);
    }

    #[test]
    fn encodes_vec3() {
        let encoded = gpu_encode(vec![], &(0.5, 0.2, 0.3)).unwrap();
        assert_eq!(encoded, words!(f32, 0.5, 0.2, 0.3, 0.0));
        assert_eq!(encoded.len(), 16);
    }

    #[test]
    fn encodes_vec4() {
        let encoded = gpu_encode(vec![], &(0.5, 0.2, 0.3, 0.8)).unwrap();
        assert_eq!(encoded, words!(f32, 0.5, 0.2, 0.3, 0.8));
        assert_eq!(encoded.len(), 16);
    }

    #[test]
    fn encodes_mat3x3() {
        let encoded =
            gpu_encode(vec![], &((0.9, 0.8, 0.7), (0.6, 0.5, 0.4), (0.3, 0.2, 0.1))).unwrap();
        assert_eq!(
            encoded,
            words!(
                f32, 0.9, 0.8, 0.7, 0.0, 0.6, 0.5, 0.4, 0.0, 0.3, 0.2, 0.1, 0.0
            )
        );
        assert_eq!(encoded.len(), 48);
    }

    #[derive(GpuLayout, GpuEncode)]
    struct SimpleStruct {
        a: u32,
        b: u32,
    }

    #[test]
    fn encodes_simple_struct() {
        let value = SimpleStruct { a: 4321, b: 1234 };
        let encoded = gpu_encode(vec![], &value).unwrap();
        assert_eq!(encoded, words!(u32, 4321, 1234));
        assert_eq!(encoded.len(), 8);
    }

    #[derive(GpuLayout, GpuEncode)]
    struct AlignedStruct {
        a: u32,
        b: (u32, u32, u32),
    }

    #[test]
    fn encodes_aligned_struct() {
        let value = AlignedStruct {
            a: 4321,
            b: (9999, 8888, 7777),
        };
        let encoded = gpu_encode(vec![], &value).unwrap();
        assert_eq!(encoded, words!(u32, 4321, 0, 0, 0, 9999, 8888, 7777, 0));
        assert_eq!(encoded.len(), 32);
    }

    #[derive(GpuLayout, GpuEncode)]
    struct PackingStruct {
        a: (u32, u32, u32),
        b: u32,
    }

    #[test]
    fn encodes_packing_struct() {
        let value = PackingStruct {
            a: (9999, 8888, 7777),
            b: 4321,
        };
        let encoded = gpu_encode(vec![], &value).unwrap();
        assert_eq!(encoded, words!(u32, 9999, 8888, 7777, 4321));
        assert_eq!(encoded.len(), 16);
    }

    #[test]
    fn decodes_u32() {
        let data = &[0x34, 0x12, 0xcd, 0xab];
        let result = gpu_decode::<u32>(Cursor::new(data)).unwrap();
        assert_eq!(result, 0xabcd1234);
    }

    #[test]
    fn decodes_vec3() {
        let data = words!(f32, 0.5, 0.2, 0.3, 0.0);
        let result = gpu_decode::<(f32, f32, f32)>(Cursor::new(data)).unwrap();
        assert_eq!(result, (0.5, 0.2, 0.3));
    }

    #[test]
    fn decodes_vec() {
        let data = words!(u32, 4321, 1234, 3412, 0, 9999, 8888, 7777, 0);
        let encoded = gpu_decode::<Vec<(u32, u32, u32)>>(Cursor::new(data)).unwrap();
        assert_eq!(encoded, vec![(4321, 1234, 3412), (9999, 8888, 7777)]);
    }
}
