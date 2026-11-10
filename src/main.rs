mod color_change;
mod concat;
mod load;
mod node;
mod output;
mod transform;
mod trim;
use crate::{
    color_change::ColorChange,
    concat::Concat,
    load::Load,
    node::Node,
    output::Output,
    transform::{Transform, TransformConfig},
    trim::Trim,
};
use cudarc::driver::CudaContext;
use cudarc::nvrtc;

fn main() {
    let cuda_ctx = CudaContext::new(0).unwrap();
    let color_change_src = include_str!("kernels/color_change.cu");
    let color_change = cuda_ctx
        .load_module(nvrtc::compile_ptx(color_change_src).unwrap())
        .unwrap()
        .load_function("color_change")
        .unwrap();
    let transform_src = include_str!("kernels/transform.cu");
    let transform = cuda_ctx
        .load_module(nvrtc::compile_ptx(transform_src).unwrap())
        .unwrap()
        .load_function("transform_macropixel")
        .unwrap();

    println!("[Log] All Cuda modules compiled");

    let v1 = Node::Load(Load::new("./media/vid2.mp4"));
    let v2 = Node::Load(Load::new("./media/vid2.mp4"));
    let v3 = Node::Load(Load::new("./media/vid2.mp4"));
    let d1 = Node::ColorChange(ColorChange::new(
        v1,
        0.1,
        color_change.clone(),
        cuda_ctx.new_stream().unwrap(),
    ));
    let d2 = Node::ColorChange(ColorChange::new(
        v2,
        0.9,
        color_change,
        cuda_ctx.new_stream().unwrap(),
    ));
    let t1 = Node::Trim(Trim::new(d2, 5, 10));
    let c1 = Node::Concat(Concat::new(vec![d1, t1]));
    // assumes a 1920x1080 clip: rotate and scale around the frame center
    let t2 = Node::Transform(Transform::new(
        v3,
        TransformConfig {
            trans_x: 100.0,
            trans_y: 50.0,
            rot_x: 960.0,
            rot_y: 540.0,
            deg_of_rot: 15.0,
            scale_x: 960.0,
            scale_y: 540.0,
            scale_factor: 0.8,
        },
        transform,
        cuda_ctx.new_stream().unwrap(),
    ));
    let c2 = Node::Concat(Concat::new(vec![c1, t2]));

    let mut o = Output::new("temp.mp4", c2);
    o.run();
}
