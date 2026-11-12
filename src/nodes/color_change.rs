use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use crate::nodes::{BlankNode, Frame, GpuFrame, HashOutput, Node, PinnedPool, Plane};
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig};
use ffmpeg_next::Rational;

pub struct ColorChange {
    pub(crate) input: Box<Node>,
    cuda_fn: CudaFunction,
    cuda_stream: Arc<CudaStream>,
    hash: HashOutput,
    frame_number: usize,
    pub(crate) strength: f32,
    // the pipeline: `ready` is handed out next, `on_gpu` is the frame after it
    ready: Option<Frame>,
    on_gpu: Option<GpuFrame>,
    primed: bool,
    pool: PinnedPool,
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
            pool: PinnedPool::new(cuda_stream.clone()),
            cuda_stream,
            strength,
            ready: None,
            on_gpu: None,
            primed: false,
        };
    }

    /// Queues the upload of one plane, the kernel on it, and the download of the result.
    /// Returns without waiting for any of it.
    /// `is_chroma` selects the U/V formula (centered on 128) over the luma one.
    fn run_plane(&mut self, plane: &[u8], is_chroma: bool) -> Plane {
        let cuda_in = self.pool.upload(plane);
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
        let result = self.pool.download(&buff);
        self.pool.keep(cuda_in);
        self.pool.keep(buff);
        return result;
    }

    /// Pulls the next input frame and sends it to the GPU. `None` when the input has ended.
    fn launch(&mut self) -> Option<GpuFrame> {
        let f = self.input.next_frame()?;
        let y = self.run_plane(&f.y, false);
        let u = self.run_plane(&f.u, true);
        let v = self.run_plane(&f.v, true);
        let a = f.a.map(Plane::Host);
        return Some(self.pool.seal(f.width, f.height, y, u, v, a));
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
        self.ready = None;
        self.on_gpu = None;
        self.primed = false;
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
