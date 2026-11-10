use crate::{color_change::ColorChange, concat::Concat, load::Load, output::Output, trim::Trim};
use ffmpeg_next::Rational;
pub type HashOutput = u64;
/// Planar YUV 4:2:2, tightly packed (no stride padding).
/// `y` is `width * height`; `u` and `v` are each `(width / 2) * height`.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}
pub trait BlankNode {
    fn next_frame(&mut self) -> Option<Frame>;
    fn reset(&mut self);
    fn node_hash(&mut self);
    fn get_hash(&self) -> HashOutput;
    /// Frame rate of the frames this node produces, as an exact fraction (e.g. 30000/1001).
    fn fps(&mut self) -> Rational;
}
pub enum Node {
    Load(Load),
    Output(Output),
    Concat(Concat),
    ColorChange(ColorChange),
    Trim(Trim),
}

impl BlankNode for Node {
    fn next_frame(&mut self) -> Option<Frame> {
        return match self {
            Self::Load(l) => l.next_frame(),
            Self::Output(o) => o.next_frame(),
            Self::Concat(c) => c.next_frame(),
            Self::ColorChange(d) => d.next_frame(),
            Self::Trim(t) => t.next_frame(),
        };
    }
    fn reset(&mut self) {
        match self {
            Self::Load(l) => l.reset(),
            Self::Output(o) => o.reset(),
            Self::Concat(c) => c.reset(),
            Self::ColorChange(d) => d.reset(),
            Self::Trim(t) => t.reset(),
        };
    }
    fn node_hash(&mut self) {
        match self {
            Self::Load(l) => l.node_hash(),
            Self::Output(o) => o.node_hash(),
            Self::Concat(c) => c.node_hash(),
            Self::ColorChange(d) => d.node_hash(),
            Self::Trim(t) => t.node_hash(),
            _ => todo!(),
        };
    }
    fn get_hash(&self) -> HashOutput {
        return match &self {
            &Self::Load(l) => l.get_hash(),
            Self::Output(o) => o.get_hash(),
            Self::Concat(c) => c.get_hash(),
            Self::ColorChange(d) => d.get_hash(),
            Self::Trim(t) => t.get_hash(),
            _ => todo!(),
        };
    }
    fn fps(&mut self) -> Rational {
        return match self {
            Self::Load(l) => l.fps(),
            Self::Output(o) => o.fps(),
            Self::Concat(c) => c.fps(),
            Self::ColorChange(d) => d.fps(),
            Self::Trim(t) => t.fps(),
        };
    }
}
