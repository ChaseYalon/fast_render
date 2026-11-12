mod nodes;
mod project;
mod interpreter;
mod fuzzer;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc;
use std::sync::Arc;
use crate::fuzzer::Fuzzer;
use crate::interpreter::GraphBuilder;
use crate::nodes::{BlankNode, Node};

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

/// Prints count, mean and median of `times_ns`. All the maths is done in
/// nanoseconds; values are only converted to seconds when printed.
fn print_stats(label: &str, times_ns: &[u128]) {
    let count = times_ns.len();
    if count == 0 {
        println!("{label}: no runs");
        return;
    }

    let mut sorted = times_ns.to_vec();
    sorted.sort_unstable();

    let sum: u128 = sorted.iter().sum();
    let mean_ns = sum as f64 / count as f64;
    let median_ns = if count % 2 == 1 {
        sorted[count / 2] as f64
    } else {
        (sorted[count / 2 - 1] + sorted[count / 2]) as f64 / 2.0
    };

    let to_secs = |ns: f64| ns / 1_000_000_000.0;
    println!("{label}");
    println!("  count  : {count}");
    println!("  mean   : {:.4} s", to_secs(mean_ns));
    println!("  median : {:.4} s", to_secs(median_ns));
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
    if args[1] == "--time" {
        let mut cold_times: Vec<u128> = vec![];
        let mut hot_times: Vec<u128> = vec![];
        let path = if args.len() > 2 && args[2] == "--abr" {
            "./vids/abridged/"
        } else {
            "./vids/full/"
        };
        for f in fs::read_dir(path).unwrap() {
            if let Ok(file) = f {
                println!("{:?}", file.path());
                let text = fs::read_to_string(file.path().clone()).unwrap();
                let builder = GraphBuilder::new();
                let (root, _) = builder.build(text, cuda_ctx.clone(), kernels.clone());
                if let Node::Output(mut o) = root {
                    let start = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("System time is before the Unix epoch")
                        .as_nanos();
                    o.run();
                    let end = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("System time is before the Unix epoch")
                        .as_nanos();
                    println!("[LOG] Cold Time: {}", end - start / 1e9 as u128);
                    cold_times.push(end - start);
                    o.reset(); // the first run consumed every frame; rewind the graph

                    let start = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("System time is before the Unix epoch")
                        .as_nanos();
                    o.run();
                    let end = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("System time is before the Unix epoch")
                        .as_nanos();
                    println!("[LOG] Hot Time: {}", start - end / 1e9 as u128);
                    hot_times.push(end - start)
                } else {
                    unreachable!()
                }
            } else {
                panic!();
            }
        }
        print_stats("Cold runs", &cold_times);
        print_stats("Hot runs", &hot_times);
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
