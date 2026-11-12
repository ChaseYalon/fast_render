use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::nodes::{BlankNode, Frame, GpuFrame, HashOutput, Node, PinnedPool};

/// Layers `top` over `bottom` using their alpha planes.
pub struct Pancake {
    pub(crate) bottom: Box<Node>,
    pub(crate) top: Box<Node>,
    hash: HashOutput,
    cuda_fn: CudaFunction,
    stream: Arc<CudaStream>,
    // the pipeline: `ready` is handed out next, `on_gpu` is the frame after it
    ready: Option<Frame>,
    on_gpu: Option<GpuFrame>,
    primed: bool,
    pool: PinnedPool,
}

impl Pancake {
    pub fn new(
        bottom: Node,
        top: Node,
        cuda_fn: CudaFunction,
        stream: Arc<CudaStream>,
    ) -> Pancake {
        return Pancake {
            bottom: Box::new(bottom),
            top: Box::new(top),
            hash: 0,
            cuda_fn,
            pool: PinnedPool::new(stream.clone()),
            stream,
            ready: None,
            on_gpu: None,
            primed: false,
        };
    }
}
impl BlankNode for Pancake {
    fn reset(&mut self) {
        self.top.reset();
        self.bottom.reset();
        self.ready = None;
        self.on_gpu = None;
        self.primed = false;
    }
    fn fps(&mut self) -> ffmpeg_next::Rational {
        let fps = self.bottom.fps();
        assert_eq!(fps, self.top.fps(), "pancake inputs must have the same fps");
        return fps;
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.bottom.node_hash();
        self.top.node_hash();
        let mut h = DefaultHasher::new();
        self.bottom.get_hash().hash(&mut h);
        self.top.get_hash().hash(&mut h);
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

impl Pancake {
    /// Pulls the next frame from both inputs and queues the upload, the kernel and the
    /// download. `None` when both inputs have ended.
    fn launch(&mut self) -> Option<GpuFrame> {
        // when one input runs out, the other plays on by itself
        let (bot, top) = match (self.bottom.next_frame(), self.top.next_frame()) {
            (Some(b), Some(t)) => (b, t),
            (Some(only), None) | (None, Some(only)) => return Some(GpuFrame::from_host(only)),
            (None, None) => return None,
        };
        assert!(
            bot.width == top.width && bot.height == top.height,
            "pancake inputs must be the same size"
        );
        let len = bot.y.len();
        let s = &self.stream;

        let bot_a = bot.a.unwrap_or_else(|| vec![255; len]);
        let top_a = top.a.unwrap_or_else(|| vec![255; len]);

        let y_bot = self.pool.upload(&bot.y);
        let u_bot = self.pool.upload(&bot.u);
        let v_bot = self.pool.upload(&bot.v);
        let a_bot = self.pool.upload(&bot_a);
        let y_top = self.pool.upload(&top.y);
        let u_top = self.pool.upload(&top.u);
        let v_top = self.pool.upload(&top.v);
        let a_top = self.pool.upload(&top_a);

        let mut out_y = s.alloc_zeros::<u8>(len).unwrap();
        let mut out_u = s.alloc_zeros::<u8>(bot.u.len()).unwrap();
        let mut out_v = s.alloc_zeros::<u8>(bot.v.len()).unwrap();
        let mut out_a = s.alloc_zeros::<u8>(len).unwrap();

        let num_pairs = (len / 2) as i32;

        let mut builder = s.launch_builder(&self.cuda_fn);
        builder
            .arg(&y_bot)
            .arg(&u_bot)
            .arg(&v_bot)
            .arg(&a_bot)
            .arg(&y_top)
            .arg(&u_top)
            .arg(&v_top)
            .arg(&a_top)
            .arg(&mut out_y)
            .arg(&mut out_u)
            .arg(&mut out_v)
            .arg(&mut out_a)
            .arg(&num_pairs);
        unsafe { builder.launch(LaunchConfig::for_num_elems(num_pairs as u32)) }.unwrap();

        let y = self.pool.download(&out_y);
        let u = self.pool.download(&out_u);
        let v = self.pool.download(&out_v);
        let a = self.pool.download(&out_a);
        for buf in [
            y_bot, u_bot, v_bot, a_bot, y_top, u_top, v_top, a_top, out_y, out_u, out_v, out_a,
        ] {
            self.pool.keep(buf);
        }
        return Some(self.pool.seal(bot.width, bot.height, y, u, v, Some(a)));
    }
}
