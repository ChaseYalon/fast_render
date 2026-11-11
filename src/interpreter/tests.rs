use std::sync::Arc;

use cudarc::driver::{CudaContext};
use ffmpeg_next::Rational;
use crate::nodes::Node;
use crate::nodes::pancake::Pancake;
use crate::{Kernels, interpreter::GraphBuilder};
fn setup() -> (GraphBuilder, Kernels, Arc<CudaContext>) {
    let ctx = CudaContext::new(0).unwrap();
    let k = Kernels::new(&ctx);
    return (GraphBuilder::new(), k, ctx);
}
#[test]
fn project_setup() {
    let (builder, k, ctx) = setup();
    let (o, p) = builder.build(r#"PROJECT(width=1920, height=1080, fps=30); l1 = LOAD(file="vid1.mp4"); OUTPUT(file="temp.mp4", input=l1);"#.to_string(), ctx, k);
    assert!(p.fps == Rational(30, 1) && p.width == 1920 && p.height == 1080);
}

#[test]
fn test1() {
    let (builder, kernels, c) = setup();
    let (o, c) = builder.build(r#"
        PROJECT(width=1920, height=1080, fps=30);
        l1=LOAD(file="vid1.mp4");
        l2=LOAD(file="vid2.mp4");
        t1=TRIM(input=l1, start=10, end=15);
        cr1=CROP(input=l2, left_x=9, left_y=90, right_x=15, right_y=140);
        c1=PANCAKE(bottom=t1, top=cr1);
        OUTPUT(file="temp.mp4", input=c1);
    "#.to_string(), c, kernels);
    if let Node::Output(o) = o {
        assert!(o.config == c);
        assert!(o.output_file_name == "temp.mp4");
        assert!(matches!(*o.input_node, Node::Pancake(_)))
    } else {
        panic!();
    }
}