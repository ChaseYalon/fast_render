use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use crate::nodes::{BlankNode, Frame, HashOutput, Node};
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig};
use ffmpeg_next::Rational;

/// Scales the color of a clip: 0.0 is grayscale, 1.0 is unchanged, 2.0 doubles the color.
pub struct SaturationChange {
    pub(crate) input: Box<Node>,
    cuda_fn: CudaFunction,
    cuda_stream: Arc<CudaStream>,
    hash: HashOutput,
    pub(crate) factor: f32,
}

impl SaturationChange {
    pub fn new(
        input: Node,
        factor: f32,
        cuda_fn: CudaFunction,
        cuda_stream: Arc<CudaStream>,
    ) -> SaturationChange {
        assert!(factor >= 0.0, "saturation factor can't be negative");
        return SaturationChange {
            input: Box::new(input),
            cuda_fn,
            cuda_stream,
            hash: 0,
            factor,
        };
    }

    fn run_plane(&self, plane: &[u8]) -> Vec<u8> {
        let cuda_in = self.cuda_stream.clone_htod(plane).unwrap();
        let mut out = self.cuda_stream.alloc_zeros::<u8>(plane.len()).unwrap();
        let n = plane.len() as i32;
        let mut builder = self.cuda_stream.launch_builder(&self.cuda_fn);
        builder
            .arg(&cuda_in)
            .arg(&self.factor)
            .arg(&n)
            .arg(&mut out);
        let cfg = LaunchConfig::for_num_elems(plane.len() as u32);
        unsafe { builder.launch(cfg) }.unwrap();
        return self.cuda_stream.clone_dtoh(&out).unwrap();
    }
}

impl BlankNode for SaturationChange {
    fn fps(&mut self) -> Rational {
        return self.input.fps();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.input.get_hash().hash(&mut h);
        self.cuda_fn.const_size_bytes().unwrap().hash(&mut h);
        self.factor.to_bits().hash(&mut h);
        self.hash = h.finish();
    }
    fn reset(&mut self) {
        self.input.reset();
    }
    fn next_frame(&mut self) -> Option<Frame> {
        let f = self.input.next_frame()?;
        let u = self.run_plane(&f.u);
        let v = self.run_plane(&f.v);
        return Some(Frame {
            width: f.width,
            height: f.height,
            y: f.y,
            u,
            v,
            a: f.a,
        });
    }
}
