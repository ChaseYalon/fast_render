use std::{hash::{DefaultHasher, Hash, Hasher}, sync::Arc};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;
use crate::node::{BlankNode, Frame, HashOutput, Node};

pub struct Darken {
    input: Box<Node>,
    cuda_fn: CudaFunction,
    cuda_stream: Arc<CudaStream>,
    hash: HashOutput,
    frame_number: usize,
    strength: f32
}

impl Darken {
    pub fn new(input: Node, strength: f32, cuda_fn: CudaFunction, cuda_stream: Arc<CudaStream>) -> Darken {
        return Darken { input: Box::new(input), cuda_fn, hash: 0, frame_number: 0, cuda_stream, strength };
    }
}
impl BlankNode for Darken {
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.input.get_hash().hash(&mut h);
        self.cuda_fn.const_size_bytes().unwrap().hash(&mut h);
        self.strength.to_bits().hash(&mut h);
        self.hash =  h.finish();
    }
    fn reset(&mut self) {
        self.input.reset();
        self.frame_number = 0;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        let previous_frame = self.input.next_frame();
        if let Some(f) = previous_frame {
            let cuda_y = self.cuda_stream.clone_htod(&f.y).unwrap();
            let mut buff = self.cuda_stream.alloc_zeros::<u8>(f.y.len()).unwrap();
            let mut builder = self.cuda_stream.launch_builder(&self.cuda_fn);
            builder.arg(&cuda_y);
            builder.arg(&self.strength);
            let rust_sucks = f.y.len() as i32;
            builder.arg(&rust_sucks);
            builder.arg(&mut buff);
            let cfg = LaunchConfig::for_num_elems(f.y.len() as u32);
            unsafe {builder.launch(cfg)}.unwrap();
            let out_y = self.cuda_stream.clone_dtoh(&buff).unwrap();
            return Some (
                Frame { width: f.width, height: f.height, y: out_y, u: f.u, v: f.v }
            );
        } else {
            return None;
        }
    }
}