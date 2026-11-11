use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::node::{BlankNode, Frame, HashOutput, Node};

/// Gaussian blur of the whole frame.
pub struct Blur {
    input: Box<Node>,
    cuda_fn: CudaFunction,
    stream: Arc<CudaStream>,
    hash: HashOutput,
    sigma: f32,
    luma_weights: CudaSlice<f32>,
    luma_radius: i32,
    chroma_weights: CudaSlice<f32>,
    chroma_radius: i32,
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
            stream,
            hash: 0,
            sigma,
            luma_weights,
            luma_radius,
            chroma_weights,
            chroma_radius,
        };
    }

    fn run_plane(
        &self,
        plane: &[u8],
        w: i32,
        h: i32,
        h_weights: &CudaSlice<f32>,
        h_radius: i32,
        v_weights: &CudaSlice<f32>,
        v_radius: i32,
    ) -> Vec<u8> {
        let s = &self.stream;
        let input = s.clone_htod(plane).unwrap();
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

        return s.clone_dtoh(&out).unwrap();
    }
}

impl BlankNode for Blur {
    fn reset(&mut self) {
        self.input.reset();
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
        let f = self.input.next_frame()?;
        let w = f.width as i32;
        let h = f.height as i32;
        let (lw, lr) = (&self.luma_weights, self.luma_radius);
        let (cw, cr) = (&self.chroma_weights, self.chroma_radius);

        let y = self.run_plane(&f.y, w, h, lw, lr, lw, lr);
        let u = self.run_plane(&f.u, w / 2, h, cw, cr, lw, lr);
        let v = self.run_plane(&f.v, w / 2, h, cw, cr, lw, lr);
        let a = f
            .a
            .as_ref()
            .map(|a| self.run_plane(a, w, h, lw, lr, lw, lr));

        return Some(Frame {
            width: f.width,
            height: f.height,
            y,
            u,
            v,
            a,
        });
    }
}
