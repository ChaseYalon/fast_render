use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use crate::node::{BlankNode, Frame, HashOutput, Node};
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig};
use ffmpeg_next::Rational;

pub struct ColorChange {
    input: Box<Node>,
    cuda_fn: CudaFunction,
    cuda_stream: Arc<CudaStream>,
    hash: HashOutput,
    frame_number: usize,
    strength: f32,
}

impl ColorChange {
    pub fn new(
        input: Node,
        strength: f32,
        cuda_fn: CudaFunction,
        cuda_stream: Arc<CudaStream>,
    ) -> ColorChange {
        return ColorChange {
            input: Box::new(input),
            cuda_fn,
            hash: 0,
            frame_number: 0,
            cuda_stream,
            strength,
        };
    }

    /// Uploads one plane, runs the kernel on it, and downloads the result.
    /// `is_chroma` selects the U/V formula (centered on 128) over the luma one.
    fn run_plane(&self, plane: &[u8], is_chroma: bool) -> Vec<u8> {
        let cuda_in = self.cuda_stream.clone_htod(plane).unwrap();
        let mut buff = self.cuda_stream.alloc_zeros::<u8>(plane.len()).unwrap();
        let n = plane.len() as i32;
        let chroma_flag = is_chroma as i32;
        let mut builder = self.cuda_stream.launch_builder(&self.cuda_fn);
        builder.arg(&cuda_in);
        builder.arg(&self.strength);
        builder.arg(&n);
        builder.arg(&mut buff);
        builder.arg(&chroma_flag);
        let cfg = LaunchConfig::for_num_elems(plane.len() as u32);
        unsafe { builder.launch(cfg) }.unwrap();
        return self.cuda_stream.clone_dtoh(&buff).unwrap();
    }
}
impl BlankNode for ColorChange {
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
        self.strength.to_bits().hash(&mut h);
        self.hash = h.finish();
    }
    fn reset(&mut self) {
        self.input.reset();
        self.frame_number = 0;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        let previous_frame = self.input.next_frame();
        if let Some(f) = previous_frame {
            let y = self.run_plane(&f.y, false);
            let u = self.run_plane(&f.u, true);
            let v = self.run_plane(&f.v, true);
            return Some(Frame {
                width: f.width,
                height: f.height,
                y,
                u,
                v,
            });
        } else {
            return None;
        }
    }
}
