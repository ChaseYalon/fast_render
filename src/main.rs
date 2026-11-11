mod nodes;
mod project;
mod interpreter;
use ffmpeg_next::Rational;
use crate::nodes::{
    blur::Blur,
    color_change::ColorChange,
    concat::Concat,
    crop::Crop,
    fps_change::FpsChange,
    load::Load,
    node::Node,
    output::Output,
    pancake::Pancake,
    saturation_change::SaturationChange,
    split::Split,
    transform::{Transform, TransformConfig},
    transition::{Transition, TransitionKind, WipeDirection},
    trim::Trim,
};
use crate::project::ProjectConfig;
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc;
use std::sync::Arc;

/// Every compiled CUDA kernel, ready to be handed to the nodes that use it.
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

    let project = ProjectConfig {
        width: 1920,
        height: 1080,
        fps: Rational(30, 1),
    };
    let v1 = Node::Load(Load::new("./media/vid2.mp4"));
    let v2 = Node::Load(Load::new("./media/vid2.mp4"));
    let v3 = Node::Load(Load::new("./media/vid1.mp4"));
    let v3 = Node::Trim(Trim::new(v3, 20, 36));
    let (v3a, v3b) = Split::new(v3, 5.0);
    let v3 = Node::Concat(Concat::new(vec![v3a, v3b]));
    let d1 = Node::ColorChange(ColorChange::new(
        v1,
        0.1,
        kernels.color_change.clone(),
        cuda_ctx.new_stream().unwrap(),
    ));
    let d2 = Node::ColorChange(ColorChange::new(
        v2,
        0.9,
        kernels.color_change,
        cuda_ctx.new_stream().unwrap(),
    ));
    let t1 = Node::Trim(Trim::new(d2, 5, 10));
    let c1 = Node::Transition(Transition::new(
        d1,
        t1,
        TransitionKind::DipToBlack,
        1,
        kernels.transition.clone(),
        cuda_ctx.new_stream().unwrap(),
    ));
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
        kernels.transform,
        cuda_ctx.new_stream().unwrap(),
    ));
    let cr1 = Node::Crop(Crop::new(c1, 320, 400, 1000, 1000));
    let v5 = Node::Load(Load::new("./media/vid3.mp4"));
    let v5 = Node::FpsChange(FpsChange::new(v5, project.fps));
    let c2 = Node::Transition(Transition::new(
        cr1,
        t2,
        TransitionKind::Crossfade,
        1,
        kernels.transition.clone(),
        cuda_ctx.new_stream().unwrap(),
    ));
    let c2 = Node::Transition(Transition::new(
        c2,
        v5,
        TransitionKind::Wipe(WipeDirection::LeftToRight),
        1,
        kernels.transition,
        cuda_ctx.new_stream().unwrap(),
    ));
    let v4 = Node::Load(Load::new("./media/vid1.mp4"));
    let v4 = Node::Trim(Trim::new(v4, 0, 40));
    let v4 = Node::Blur(Blur::new(v4, 8.0, kernels.blur, cuda_ctx.new_stream().unwrap()));
    let v4 = Node::SaturationChange(SaturationChange::new(
        v4,
        0.0,
        kernels.saturation,
        cuda_ctx.new_stream().unwrap(),
    ));
    let c2 = Node::Pancake(Pancake::new(
        v4,
        c2,
        kernels.pancake,
        cuda_ctx.new_stream().unwrap(),
    ));


    let mut o = Output::new("temp.mp4", c2, project);
    o.run();
}
