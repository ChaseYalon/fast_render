use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use crate::nodes::{BlankNode, Frame, GpuFrame, HashOutput, Node, PinnedPool, Plane};
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
    // the pipeline: `ready` is handed out next, `on_gpu` is the frame after it
    ready: Option<Frame>,
    on_gpu: Option<GpuFrame>,
    primed: bool,
    pool: PinnedPool,
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
            pool: PinnedPool::new(cuda_stream.clone()),
            cuda_stream,
            hash: 0,
            factor,
            ready: None,
            on_gpu: None,
            primed: false,
        };
    }

    /// Queues the upload of one plane, the kernel on it, and the download of the result.
    /// Returns without waiting for any of it.
    fn run_plane(&mut self, plane: &[u8]) -> Plane {
        let cuda_in = self.pool.upload(plane);
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
        let result = self.pool.download(&out);
        self.pool.keep(cuda_in);
        self.pool.keep(out);
        return result;
    }

    /// Pulls the next input frame and sends it to the GPU. `None` when the input has ended.
    fn launch(&mut self) -> Option<GpuFrame> {
        let f = self.input.next_frame()?;
        let u = self.run_plane(&f.u);
        let v = self.run_plane(&f.v);
        // saturation only touches the colour planes, so y and alpha never leave the CPU
        let y = Plane::Host(f.y);
        let a = f.a.map(Plane::Host);
        return Some(self.pool.seal(f.width, f.height, y, u, v, a));
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
