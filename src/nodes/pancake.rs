use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::nodes::{BlankNode, Frame, HashOutput, Node};

/// Layers `top` over `bottom` using their alpha planes.
pub struct Pancake {
    pub(crate) bottom: Box<Node>,
    pub(crate) top: Box<Node>,
    hash: HashOutput,
    cuda_fn: CudaFunction,
    stream: Arc<CudaStream>,
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
            stream,
        };
    }
}
impl BlankNode for Pancake {
    fn reset(&mut self) {
        self.top.reset();
        self.bottom.reset();
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
        // when one input runs out, the other plays on by itself
        let (bot, top) = match (self.bottom.next_frame(), self.top.next_frame()) {
            (Some(b), Some(t)) => (b, t),
            (Some(only), None) | (None, Some(only)) => return Some(only),
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

        let y_bot = s.clone_htod(&bot.y).unwrap();
        let u_bot = s.clone_htod(&bot.u).unwrap();
        let v_bot = s.clone_htod(&bot.v).unwrap();
        let a_bot = s.clone_htod(&bot_a).unwrap();
        let y_top = s.clone_htod(&top.y).unwrap();
        let u_top = s.clone_htod(&top.u).unwrap();
        let v_top = s.clone_htod(&top.v).unwrap();
        let a_top = s.clone_htod(&top_a).unwrap();

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

        return Some(Frame {
            width: bot.width,
            height: bot.height,
            y: s.clone_dtoh(&out_y).unwrap(),
            u: s.clone_dtoh(&out_u).unwrap(),
            v: s.clone_dtoh(&out_v).unwrap(),
            a: Some(s.clone_dtoh(&out_a).unwrap()),
        });
    }
}
