use std::{hash::DefaultHasher, sync::Arc};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::nodes::{BlankNode, Frame, GpuFrame, HashOutput, Node, PinnedPool};
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
    pub(crate) input: Box<Node>,
    cuda_fn: CudaFunction,
    cuda_stream: Arc<CudaStream>,
    hash: HashOutput,
    pub(crate) config: TransformConfig,
    // the pipeline: `ready` is handed out next, `on_gpu` is the frame after it
    ready: Option<Frame>,
    on_gpu: Option<GpuFrame>,
    primed: bool,
    pool: PinnedPool,
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
            pool: PinnedPool::new(cuda_stream.clone()),
            cuda_stream,
            hash: 0,
            config,
            ready: None,
            on_gpu: None,
            primed: false,
        };
    }
}

impl BlankNode for Transform {
    fn reset(&mut self) {
        self.input.reset();
        self.ready = None;
        self.on_gpu = None;
        self.primed = false;
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
        if !self.primed {
            // the first call has nothing waiting, so it pays for a full frame
            self.ready = self.launch().map(|g| self.pool.finish(g));
            self.on_gpu = self.launch();
            self.primed = true;
        }
        let out = self.ready.take()?; // 1. hand out the frame that is already here
        self.ready = self.on_gpu.take().map(|g| self.pool.finish(g)); // 2. collect the next one
        self.on_gpu = self.launch(); // 3. send the one after that to the GPU
        return Some(out);
    }
}

impl Transform {
    /// Pulls the next input frame and queues its upload, the kernel and the download.
    /// `None` when the input has ended.
    fn launch(&mut self) -> Option<GpuFrame> {
        {
            let f = self.input.next_frame()?;
            let in_y = self.pool.upload(&f.y);
            let in_u = self.pool.upload(&f.u);
            let in_v = self.pool.upload(&f.v);

            // with no alpha plane the kernel never reads in_a, so in_y stands in for it
            let in_a_buf = f.a.as_ref().map(|a| self.pool.upload(a));
            let in_a = in_a_buf.as_ref().unwrap_or(&in_y);
            let has_alpha = f.a.is_some() as i32;

            let mut out_y = self.cuda_stream.alloc_zeros::<u8>(f.y.len()).unwrap();
            let mut out_u = self.cuda_stream.alloc_zeros::<u8>(f.u.len()).unwrap();
            let mut out_v = self.cuda_stream.alloc_zeros::<u8>(f.v.len()).unwrap();
            let mut out_a = self.cuda_stream.alloc_zeros::<u8>(f.y.len()).unwrap();

            let width = f.width as i32;
            let height = f.height as i32;
            let c = &self.config;

            let mut builder = self.cuda_stream.launch_builder(&self.cuda_fn);
            builder
                .arg(&in_y)
                .arg(&in_u)
                .arg(&in_v)
                .arg(in_a)
                .arg(&mut out_y)
                .arg(&mut out_u)
                .arg(&mut out_v)
                .arg(&mut out_a)
                .arg(&width)
                .arg(&height)
                .arg(&has_alpha)
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

            let y = self.pool.download(&out_y);
            let u = self.pool.download(&out_u);
            let v = self.pool.download(&out_v);
            let a = self.pool.download(&out_a);
            for buf in [in_y, in_u, in_v, out_y, out_u, out_v, out_a] {
                self.pool.keep(buf);
            }
            if let Some(buf) = in_a_buf {
                self.pool.keep(buf);
            }
            return Some(self.pool.seal(f.width, f.height, y, u, v, Some(a)));
        }
    }
}
