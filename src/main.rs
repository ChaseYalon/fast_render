mod nodes;
mod project;
mod interpreter;
mod fuzzer;
use std::{env, fs};
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc;
use std::sync::Arc;

use crate::fuzzer::Fuzzer;
use crate::interpreter::GraphBuilder;
use crate::nodes::Node;

#[derive(Clone)]
pub struct Kernels {
    pub color_change: CudaFunction,
    pub transform: CudaFunction,
    pub pancake: CudaFunction,
    pub blur: CudaFunction,
    pub saturation: CudaFunction,
    pub transition: CudaFunction,
}

impl Kernels {
    /// Compiles the CUDA source `src` and loads its function `name` onto `ctx`.
    fn load(ctx: &Arc<CudaContext>, src: &str, name: &str) -> CudaFunction {
        ctx.load_module(nvrtc::compile_ptx(src).unwrap())
            .unwrap()
            .load_function(name)
            .unwrap()
    }

    pub fn new(ctx: &Arc<CudaContext>) -> Kernels {
        let kernels = Kernels {
            color_change: Self::load(ctx, include_str!("kernels/color_change.cu"), "color_change"),
            transform: Self::load(ctx, include_str!("kernels/transform.cu"), "transform_macropixel"),
            pancake: Self::load(ctx, include_str!("kernels/pancake.cu"), "pancake"),
            blur: Self::load(ctx, include_str!("kernels/blur.cu"), "blur"),
            saturation: Self::load(
                ctx,
                include_str!("kernels/saturation_change.cu"),
                "saturation_change",
            ),
            transition: Self::load(ctx, include_str!("kernels/transition.cu"), "transition"),
        };
        println!("[Log] All Cuda modules compiled");
        kernels
    }
}

fn main() {
    let cuda_ctx = CudaContext::new(0).unwrap();
    let kernels = Kernels::new(&cuda_ctx);
    let args: Vec<String> = env::args().collect();

    if args[1] == "--fuzz" {
        let n: usize = args[2].parse().unwrap();
        for _ in 0..n {
            let mut f = Fuzzer::new();
            f.fuzz(true);
        }
        return;
    }

    let text = fs::read_to_string(args[1].clone()).unwrap();
    let builder = GraphBuilder::new();
    let (root, _) = builder.build(text, cuda_ctx, kernels);
    if let Node::Output(mut o) = root {
        o.run();
    } else {
        unreachable!()
    }
}
