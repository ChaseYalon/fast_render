use std::{
    collections::VecDeque,
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use ffmpeg_next::Rational;

use crate::node::{BlankNode, Frame, HashOutput, Node};

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
    a: Box<Node>,
    b: Box<Node>,
    kind: TransitionKind,
    seconds: usize,
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
            stream,
            hash: 0,
            queue: VecDeque::new(),
            primed: false,
            a_done: false,
            blend_total: None,
            blend_index: 0,
        };
    }

    fn blend(&self, a: Frame, b: Frame, progress: f32) -> Frame {
        assert!(
            a.width == b.width && a.height == b.height,
            "transition inputs must be the same size"
        );
        let len = a.y.len();
        let s = &self.stream;
        let a_alpha = a.a.unwrap_or_else(|| vec![255; len]);
        let b_alpha = b.a.unwrap_or_else(|| vec![255; len]);

        let y_a = s.clone_htod(&a.y).unwrap();
        let u_a = s.clone_htod(&a.u).unwrap();
        let v_a = s.clone_htod(&a.v).unwrap();
        let a_a = s.clone_htod(&a_alpha).unwrap();
        let y_b = s.clone_htod(&b.y).unwrap();
        let u_b = s.clone_htod(&b.u).unwrap();
        let v_b = s.clone_htod(&b.v).unwrap();
        let a_b = s.clone_htod(&b_alpha).unwrap();

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

        return Frame {
            width: a.width,
            height: a.height,
            y: s.clone_dtoh(&out_y).unwrap(),
            u: s.clone_dtoh(&out_u).unwrap(),
            v: s.clone_dtoh(&out_v).unwrap(),
            a: Some(s.clone_dtoh(&out_a).unwrap()),
        };
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
    }
    fn next_frame(&mut self) -> Option<Frame> {
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
                        return self.queue.pop_front();
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
                None => return Some(a_frame),
            },
            None => return self.b.next_frame(),
        }
    }
}
