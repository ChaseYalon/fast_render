mod color_change;
mod concat;
mod load;
mod node;
mod output;
mod project;
mod transform;
mod crop;
mod trim;
mod pancake;
mod blur;
mod saturation_change;
use ffmpeg_next::Rational;
use crate::{
    blur::Blur,
    color_change::ColorChange,
    concat::Concat,
    crop::Crop,
    load::Load,
    node::Node,
    output::Output,
    pancake::Pancake,
    project::ProjectConfig,
    saturation_change::SaturationChange,
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
    let pancake_src = include_str!("kernels/pancake.cu");
    let pancake = cuda_ctx
        .load_module(nvrtc::compile_ptx(pancake_src).unwrap())
        .unwrap()
        .load_function("pancake")
        .unwrap();

    let blur_src = include_str!("kernels/blur.cu");
    let blur = cuda_ctx
        .load_module(nvrtc::compile_ptx(blur_src).unwrap())
        .unwrap()
        .load_function("blur")
        .unwrap();

    let saturation_src = include_str!("kernels/saturation_change.cu");
    let saturation = cuda_ctx
        .load_module(nvrtc::compile_ptx(saturation_src).unwrap())
        .unwrap()
        .load_function("saturation_change")
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
    let cr1 = Node::Crop(Crop::new(c1, 320, 400, 1000, 1000));
    let c2 = Node::Concat(Concat::new(vec![cr1, t2]));
    // the concatenated clips are layered over a plain clip
    let v4 = Node::Load(Load::new("./media/vid2.mp4"));
    let v4 = Node::Blur(Blur::new(v4, 8.0, blur, cuda_ctx.new_stream().unwrap()));
    let v4 = Node::SaturationChange(SaturationChange::new(
        v4,
        0.0,
        saturation,
        cuda_ctx.new_stream().unwrap(),
    ));
    let c2 = Node::Pancake(Pancake::new(
        v4,
        c2,
        pancake,
        cuda_ctx.new_stream().unwrap(),
    ));

    // every frame has to match this canvas and fps; adjust it to match the clips
    let project = ProjectConfig {
        width: 1920,
        height: 1080,
        fps: Rational(30, 1),
    };

    let mut o = Output::new("temp.mp4", c2, project);
    o.run();
}
