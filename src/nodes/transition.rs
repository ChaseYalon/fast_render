use std::{
    collections::VecDeque,
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use ffmpeg_next::Rational;

use crate::nodes::{BlankNode, Frame, GpuFrame, HashOutput, Node, PinnedPool};

#[derive(Clone, Copy, Hash)]
pub enum WipeDirection {
    LeftToRight,
    RightToLeft,
}

#[derive(Clone, Copy, Hash)]
pub enum TransitionKind {
    Crossfade,
    Wipe(WipeDirection),
    DipToBlack,
}

/// Plays `a`, then `b`, blending the last `seconds` of `a` into the first `seconds` of `b`.
/// The result is `seconds` shorter than the two clips back to back.
pub struct Transition {
    pub(crate) a: Box<Node>,
    pub(crate) b: Box<Node>,
    pub(crate) kind: TransitionKind,
    pub(crate) seconds: usize,
    cuda_fn: CudaFunction,
    stream: Arc<CudaStream>,
    hash: HashOutput,
    // the last frames of `a` that haven't been shown yet
    queue: VecDeque<Frame>,
    primed: bool,
    a_done: bool,
    // number of frames in the overlap, known once `a` has ended
    blend_total: Option<usize>,
    blend_index: usize,
    // the pipeline: `ready` is handed out next, `on_gpu` is the frame after it.
    // (`primed` above is about the queue of `a` frames, not about this.)
    ready: Option<Frame>,
    on_gpu: Option<GpuFrame>,
    pipeline_primed: bool,
    pool: PinnedPool,
}

impl Transition {
    pub fn new(
        a: Node,
        b: Node,
        kind: TransitionKind,
        seconds: usize,
        cuda_fn: CudaFunction,
        stream: Arc<CudaStream>,
    ) -> Transition {
        return Transition {
            a: Box::new(a),
            b: Box::new(b),
            kind,
            seconds,
            cuda_fn,
            pool: PinnedPool::new(stream.clone()),
            stream,
            ready: None,
            on_gpu: None,
            pipeline_primed: false,
            hash: 0,
            queue: VecDeque::new(),
            primed: false,
            a_done: false,
            blend_total: None,
            blend_index: 0,
        };
    }

    /// Queues the upload of both frames, the blend kernel and the download of the result.
    fn blend(&mut self, a: Frame, b: Frame, progress: f32) -> GpuFrame {
        assert!(
            a.width == b.width && a.height == b.height,
            "transition inputs must be the same size"
        );
        let len = a.y.len();
        let s = &self.stream;
        let a_alpha = a.a.unwrap_or_else(|| vec![255; len]);
        let b_alpha = b.a.unwrap_or_else(|| vec![255; len]);

        let y_a = self.pool.upload(&a.y);
        let u_a = self.pool.upload(&a.u);
        let v_a = self.pool.upload(&a.v);
        let a_a = self.pool.upload(&a_alpha);
        let y_b = self.pool.upload(&b.y);
        let u_b = self.pool.upload(&b.u);
        let v_b = self.pool.upload(&b.v);
        let a_b = self.pool.upload(&b_alpha);

        let mut out_y = s.alloc_zeros::<u8>(len).unwrap();
        let mut out_u = s.alloc_zeros::<u8>(a.u.len()).unwrap();
        let mut out_v = s.alloc_zeros::<u8>(a.v.len()).unwrap();
        let mut out_a = s.alloc_zeros::<u8>(len).unwrap();

        let num_pairs = (len / 2) as i32;
        let width = a.width as i32;
        let (kind, reverse) = match self.kind {
            TransitionKind::Crossfade => (0i32, 0i32),
            TransitionKind::Wipe(WipeDirection::LeftToRight) => (1, 0),
            TransitionKind::Wipe(WipeDirection::RightToLeft) => (1, 1),
            TransitionKind::DipToBlack => (2, 0),
        };

        let mut builder = s.launch_builder(&self.cuda_fn);
        builder
            .arg(&y_a)
            .arg(&u_a)
            .arg(&v_a)
            .arg(&a_a)
            .arg(&y_b)
            .arg(&u_b)
            .arg(&v_b)
            .arg(&a_b)
            .arg(&mut out_y)
            .arg(&mut out_u)
            .arg(&mut out_v)
            .arg(&mut out_a)
            .arg(&num_pairs)
            .arg(&width)
            .arg(&kind)
            .arg(&reverse)
            .arg(&progress);
        unsafe { builder.launch(LaunchConfig::for_num_elems(num_pairs as u32)) }.unwrap();

        let y = self.pool.download(&out_y);
        let u = self.pool.download(&out_u);
        let v = self.pool.download(&out_v);
        let alpha = self.pool.download(&out_a);
        for buf in [
            y_a, u_a, v_a, a_a, y_b, u_b, v_b, a_b, out_y, out_u, out_v, out_a,
        ] {
            self.pool.keep(buf);
        }
        return self.pool.seal(a.width, a.height, y, u, v, Some(alpha));
    }

    /// Works out the next output frame and queues any GPU work it needs.
    /// `None` when both inputs have ended.
    fn launch(&mut self) -> Option<GpuFrame> {
        if !self.a_done {
            if !self.primed {
                let fps = self.fps();
                let overlap =
                    self.seconds * fps.numerator() as usize / fps.denominator() as usize;
                while self.queue.len() < overlap {
                    match self.a.next_frame() {
                        Some(f) => self.queue.push_back(f),
                        None => {
                            self.a_done = true;
                            break;
                        }
                    }
                }
                self.primed = true;
            }
            // `a` is shown with a delay, so its last frames are still queued when it ends
            if !self.a_done {
                match self.a.next_frame() {
                    Some(f) => {
                        self.queue.push_back(f);
                        return self.queue.pop_front().map(GpuFrame::from_host);
                    }
                    None => self.a_done = true,
                }
            }
        }
        let total = *self.blend_total.get_or_insert(self.queue.len());

        match self.queue.pop_front() {
            Some(a_frame) => match self.b.next_frame() {
                Some(b_frame) => {
                    self.blend_index += 1;
                    let progress = self.blend_index as f32 / (total + 1) as f32;
                    return Some(self.blend(a_frame, b_frame, progress));
                }
                // `b` ended inside the overlap, so the rest of `a` plays unblended
                None => return Some(GpuFrame::from_host(a_frame)),
            },
            None => return self.b.next_frame().map(GpuFrame::from_host),
        }
    }
}

impl BlankNode for Transition {
    fn fps(&mut self) -> Rational {
        let fps = self.a.fps();
        assert_eq!(fps, self.b.fps(), "transition inputs must have the same fps");
        return fps;
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.a.node_hash();
        self.b.node_hash();
        let mut h = DefaultHasher::new();
        self.a.get_hash().hash(&mut h);
        self.b.get_hash().hash(&mut h);
        self.kind.hash(&mut h);
        self.seconds.hash(&mut h);
        self.cuda_fn.const_size_bytes().unwrap().hash(&mut h);
        self.hash = h.finish();
    }
    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
        self.queue.clear();
        self.primed = false;
        self.a_done = false;
        self.blend_total = None;
        self.blend_index = 0;
        self.ready = None;
        self.on_gpu = None;
        self.pipeline_primed = false;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        if !self.pipeline_primed {
            // the first call has nothing waiting, so it pays for a full frame
            self.ready = self.launch().map(|g| self.pool.finish(g));
            self.on_gpu = self.launch();
            self.pipeline_primed = true;
        }
        let out = self.ready.take()?; // 1. hand out the frame that is already here
        self.ready = self.on_gpu.take().map(|g| self.pool.finish(g)); // 2. collect the next one
        self.on_gpu = self.launch(); // 3. send the one after that to the GPU
        return Some(out);
    }
}
