use std::{hash::DefaultHasher, sync::Arc};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::node::{BlankNode, Frame, HashOutput, Node};
use std::hash::{Hash, Hasher};
pub struct TransformConfig {
    pub trans_x: f32,
    pub trans_y: f32,
    pub rot_x: f32,
    pub rot_y: f32,
    pub deg_of_rot: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_factor: f32,
}
impl Hash for TransformConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.trans_x.to_bits().hash(state);
        self.trans_y.to_bits().hash(state);
        self.rot_x.to_bits().hash(state);
        self.rot_y.to_bits().hash(state);
        self.deg_of_rot.to_bits().hash(state);
        self.scale_x.to_bits().hash(state);
        self.scale_y.to_bits().hash(state);
        self.scale_factor.to_bits().hash(state);
    }
}
pub struct Transform {
    input: Box<Node>,
    cuda_fn: CudaFunction,
    cuda_stream: Arc<CudaStream>,
    hash: HashOutput,
    config: TransformConfig,
}

impl Transform {
    pub fn new(
        input: Node,
        config: TransformConfig,
        cuda_fn: CudaFunction,
        cuda_stream: Arc<CudaStream>,
    ) -> Transform {
        assert!(config.scale_factor != 0.0, "scale_factor can't be 0");
        return Transform {
            input: Box::new(input),
            cuda_fn,
            cuda_stream,
            hash: 0,
            config,
        };
    }
}

impl BlankNode for Transform {
    fn reset(&mut self) {
        self.input.reset();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.config.hash(&mut h);
        self.input.get_hash().hash(&mut h);
        self.cuda_fn.const_size_bytes().unwrap().hash(&mut h);
        self.hash = h.finish();
    }
    fn fps(&mut self) -> ffmpeg_next::Rational {
        return self.input.fps();
    }
    fn next_frame(&mut self) -> Option<Frame> {
        if let Some(f) = self.input.next_frame() {
            let in_y = self.cuda_stream.clone_htod(&f.y).unwrap();
            let in_u = self.cuda_stream.clone_htod(&f.u).unwrap();
            let in_v = self.cuda_stream.clone_htod(&f.v).unwrap();

            let mut out_y = self.cuda_stream.alloc_zeros::<u8>(f.y.len()).unwrap();
            let mut out_u = self.cuda_stream.alloc_zeros::<u8>(f.u.len()).unwrap();
            let mut out_v = self.cuda_stream.alloc_zeros::<u8>(f.v.len()).unwrap();

            let width = f.width as i32;
            let height = f.height as i32;
            let c = &self.config;

            let mut builder = self.cuda_stream.launch_builder(&self.cuda_fn);
            builder
                .arg(&in_y)
                .arg(&in_u)
                .arg(&in_v)
                .arg(&mut out_y)
                .arg(&mut out_u)
                .arg(&mut out_v)
                .arg(&width)
                .arg(&height)
                .arg(&c.trans_x)
                .arg(&c.trans_y)
                .arg(&c.rot_x)
                .arg(&c.rot_y)
                .arg(&c.deg_of_rot)
                .arg(&c.scale_x)
                .arg(&c.scale_y)
                .arg(&c.scale_factor);

            // one thread per macropixel: width / 2 columns by height rows
            let cfg = LaunchConfig {
                grid_dim: ((f.width / 2 + 15) / 16, (f.height + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }.unwrap();

            let y = self.cuda_stream.clone_dtoh(&out_y).unwrap();
            let u = self.cuda_stream.clone_dtoh(&out_u).unwrap();
            let v = self.cuda_stream.clone_dtoh(&out_v).unwrap();
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
