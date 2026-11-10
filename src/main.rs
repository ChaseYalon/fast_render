mod concat;
mod load;
mod node;
mod output;
mod darken;
use cudarc::driver::CudaContext;
use cudarc::nvrtc;
use crate::{
    concat::Concat, darken::Darken, load::Load, node::Node, output::Output,
};

fn main() {
    let cuda_ctx = CudaContext::new(0).unwrap();
    let darken_src = include_str!("kernels/darken.cu");
    let darken = cuda_ctx.load_module(nvrtc::compile_ptx(darken_src).unwrap()).unwrap().load_function("darken").unwrap();

    println!("[Log] All Cuda modules compiled");
    
    let v1 = Node::Load(Load::new("./media/vid2.mp4"));
    let v2 = Node::Load(Load::new("./media/vid2.mp4"));
    let d1 = Node::Darken(Darken::new(v1, 0.2, darken, cuda_ctx.new_stream().unwrap()));
    let c = Node::Concat(Concat::new(vec![d1, v2]));
    let mut o = Output::new("temp.mp4", c);
    o.run();
}
