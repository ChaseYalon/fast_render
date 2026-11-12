use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::nodes::{BlankNode, Frame, GpuFrame, HashOutput, Node, PinnedPool, Plane};

/// Gaussian blur of the whole frame.
pub struct Blur {
    pub(crate) input: Box<Node>,
    cuda_fn: CudaFunction,
    stream: Arc<CudaStream>,
    hash: HashOutput,
    pub(crate) sigma: f32,
    luma_weights: CudaSlice<f32>,
    luma_radius: i32,
    chroma_weights: CudaSlice<f32>,
    chroma_radius: i32,
    // the pipeline: `ready` is handed out next, `on_gpu` is the frame after it
    ready: Option<Frame>,
    on_gpu: Option<GpuFrame>,
    primed: bool,
    pool: PinnedPool,
}

/// Normalized gaussian weights for offsets -radius..=radius.
fn gaussian(sigma: f32) -> (Vec<f32>, i32) {
    let radius = (3.0 * sigma).ceil() as i32;
    let mut weights: Vec<f32> = (-radius..=radius)
        .map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= total;
    }
    return (weights, radius);
}

impl Blur {
    pub fn new(input: Node, sigma: f32, cuda_fn: CudaFunction, stream: Arc<CudaStream>) -> Blur {
        assert!(sigma > 0.0, "sigma must be positive");
        let (luma, luma_radius) = gaussian(sigma);
        // chroma is half as wide, so the horizontal blur uses half the sigma
        let (chroma, chroma_radius) = gaussian(sigma / 2.0);
        let luma_weights = stream.clone_htod(&luma).unwrap();
        let chroma_weights = stream.clone_htod(&chroma).unwrap();
        return Blur {
            input: Box::new(input),
            cuda_fn,
            pool: PinnedPool::new(stream.clone()),
            stream,
            hash: 0,
            sigma,
            luma_weights,
            luma_radius,
            chroma_weights,
            chroma_radius,
            ready: None,
            on_gpu: None,
            primed: false,
        };
    }

    /// Queues the upload of one plane, both blur passes on it, and the download of the
    /// result. Returns without waiting for any of it.
    /// `chroma` planes are half as wide, so their horizontal pass uses the chroma weights.
    fn run_plane(&mut self, plane: &[u8], w: i32, h: i32, chroma: bool) -> Plane {
        let s = &self.stream;
        let (h_weights, h_radius) = if chroma {
            (&self.chroma_weights, self.chroma_radius)
        } else {
            (&self.luma_weights, self.luma_radius)
        };
        let (v_weights, v_radius) = (&self.luma_weights, self.luma_radius);
        let input = self.pool.upload(plane);
        let mut temp = s.alloc_zeros::<u8>(plane.len()).unwrap();
        let mut out = s.alloc_zeros::<u8>(plane.len()).unwrap();
        let cfg = LaunchConfig::for_num_elems(plane.len() as u32);
        let horizontal = 1i32;
        let vertical = 0i32;

        let mut builder = s.launch_builder(&self.cuda_fn);
        builder
            .arg(&input)
            .arg(&mut temp)
            .arg(h_weights)
            .arg(&w)
            .arg(&h)
            .arg(&h_radius)
            .arg(&horizontal);
        unsafe { builder.launch(cfg) }.unwrap();

        let mut builder = s.launch_builder(&self.cuda_fn);
        builder
            .arg(&temp)
            .arg(&mut out)
            .arg(v_weights)
            .arg(&w)
            .arg(&h)
            .arg(&v_radius)
            .arg(&vertical);
        unsafe { builder.launch(cfg) }.unwrap();

        let result = self.pool.download(&out);
        self.pool.keep(input);
        self.pool.keep(temp);
        self.pool.keep(out);
        return result;
    }

    /// Pulls the next input frame and sends it to the GPU. `None` when the input has ended.
    fn launch(&mut self) -> Option<GpuFrame> {
        let f = self.input.next_frame()?;
        let w = f.width as i32;
        let h = f.height as i32;

        let y = self.run_plane(&f.y, w, h, false);
        let u = self.run_plane(&f.u, w / 2, h, true);
        let v = self.run_plane(&f.v, w / 2, h, true);
        let a = f.a.as_ref().map(|a| self.run_plane(a, w, h, false));
        return Some(self.pool.seal(f.width, f.height, y, u, v, a));
    }
}

impl BlankNode for Blur {
    fn reset(&mut self) {
        self.input.reset();
        self.ready = None;
        self.on_gpu = None;
        self.primed = false;
    }
    fn fps(&mut self) -> ffmpeg_next::Rational {
        return self.input.fps();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.input.get_hash().hash(&mut h);
        self.sigma.to_bits().hash(&mut h);
        self.cuda_fn.const_size_bytes().unwrap().hash(&mut h);
        self.hash = h.finish();
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
