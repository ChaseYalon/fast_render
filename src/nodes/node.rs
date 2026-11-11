use crate::nodes::{
    blur::Blur, color_change::ColorChange, concat::Concat, crop::Crop, fps_change::FpsChange, load::Load, output::Output, pancake::Pancake, saturation_change::SaturationChange, split::Split, transform::Transform, transition::Transition, trim::Trim,
};
use ffmpeg_next::Rational;
pub type HashOutput = u64;
/// Planar YUV 4:2:2, tightly packed (no stride padding).
/// `y` is `width * height`; `u` and `v` are each `(width / 2) * height`.
/// `a` is an optional straight (not premultiplied) alpha plane of `width * height`:
/// 0 is transparent, 255 is opaque, and `None` means the whole frame is opaque.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub a: Option<Vec<u8>>,
}
pub trait BlankNode {
    fn next_frame(&mut self) -> Option<Frame>;
    fn reset(&mut self);
    fn node_hash(&mut self);
    fn get_hash(&self) -> HashOutput;
    fn fps(&mut self) -> Rational;
}
pub enum Node {
    Load(Load),
    Output(Output),
    Concat(Concat),
    ColorChange(ColorChange),
    Trim(Trim),
    Transform(Transform),
    Crop(Crop),
    Pancake(Pancake),
    Blur(Blur),
    SaturationChange(SaturationChange),
    Split(Split),
    FpsChange(FpsChange),
    Transition(Transition),
}

impl BlankNode for Node {
    fn next_frame(&mut self) -> Option<Frame> {
        return match self {
            Self::Load(l) => l.next_frame(),
            Self::Output(o) => o.next_frame(),
            Self::Concat(c) => c.next_frame(),
            Self::ColorChange(d) => d.next_frame(),
            Self::Trim(t) => t.next_frame(),
            Self::Transform(t) => t.next_frame(),
            Self::Crop(c) => c.next_frame(),
            Self::Pancake(p) => p.next_frame(),
            Self::Blur(b) => b.next_frame(),
            Self::SaturationChange(s) => s.next_frame(),
            Self::Split(s) => s.next_frame(),
            Self::FpsChange(f) => f.next_frame(),
            Self::Transition(t) => t.next_frame(),
        };
    }
    fn reset(&mut self) {
        match self {
            Self::Load(l) => l.reset(),
            Self::Output(o) => o.reset(),
            Self::Concat(c) => c.reset(),
            Self::ColorChange(d) => d.reset(),
            Self::Trim(t) => t.reset(),
            Self::Transform(t) => t.reset(),
            Self::Crop(c) => c.reset(),
            Self::Pancake(p) => p.reset(),
            Self::Blur(b) => b.reset(),
            Self::SaturationChange(s) => s.reset(),
            Self::Split(s) => s.reset(),
            Self::FpsChange(f) => f.reset(),
            Self::Transition(t) => t.reset(),
        };
    }
    fn node_hash(&mut self) {
        match self {
            Self::Load(l) => l.node_hash(),
            Self::Output(o) => o.node_hash(),
            Self::Concat(c) => c.node_hash(),
            Self::ColorChange(d) => d.node_hash(),
            Self::Trim(t) => t.node_hash(),
            Self::Transform(t) => t.node_hash(),
            Self::Crop(c) => c.node_hash(),
            Self::Pancake(p) => p.node_hash(),
            Self::Blur(b) => b.node_hash(),
            Self::SaturationChange(s) => s.node_hash(),
            Self::Split(s) => s.node_hash(),
            Self::FpsChange(f) => f.node_hash(),
            Self::Transition(t) => t.node_hash(),
        };
    }
    fn get_hash(&self) -> HashOutput {
        return match &self {
            &Self::Load(l) => l.get_hash(),
            Self::Output(o) => o.get_hash(),
            Self::Concat(c) => c.get_hash(),
            Self::ColorChange(d) => d.get_hash(),
            Self::Trim(t) => t.get_hash(),
            Self::Transform(t) => t.get_hash(),
            Self::Crop(c) => c.get_hash(),
            Self::Pancake(p) => p.get_hash(),
            Self::Blur(b) => b.get_hash(),
            Self::SaturationChange(s) => s.get_hash(),
            Self::Split(s) => s.get_hash(),
            Self::FpsChange(f) => f.get_hash(),
            Self::Transition(t) => t.get_hash(),
        };
    }
    fn fps(&mut self) -> Rational {
        return match self {
            Self::Load(l) => l.fps(),
            Self::Output(o) => o.fps(),
            Self::Concat(c) => c.fps(),
            Self::ColorChange(d) => d.fps(),
            Self::Trim(t) => t.fps(),
            Self::Transform(t) => t.fps(),
            Self::Crop(c) => c.fps(),
            Self::Pancake(p) => p.fps(),
            Self::Blur(b) => b.fps(),
            Self::SaturationChange(s) => s.fps(),
            Self::Split(s) => s.fps(),
            Self::FpsChange(f) => f.fps(),
            Self::Transition(t) => t.fps(),
        };
    }
}
