use crate::{concat::Concat, load::Load, output::Output};
pub type HashOutput = u64;
/// Length -> Width -> rgba
pub type Frame = Vec<Vec<[u8; 4]>>;
pub trait BlankNode {
    fn next_frame(&mut self) -> Option<Frame>;
    fn reset(&mut self);
    fn node_hash(&mut self);
    fn get_hash(&self) -> HashOutput;
}
pub enum Node {
    Load(Load),
    Output(Output),
    Concat(Concat),
}

impl BlankNode for Node {
    fn next_frame(&mut self) -> Option<Frame> {
        return match self {
            Self::Load(l) => l.next_frame(),
            Self::Output(o) => o.next_frame(),
            Self::Concat(c) => c.next_frame(),
        };
    }
    fn reset(&mut self) {
        match self {
            Self::Load(l) => l.reset(),
            Self::Output(o) => o.reset(),
            Self::Concat(c) => c.reset(),
        };
    }
    fn node_hash(&mut self) {
        match self {
            Self::Load(l) => l.node_hash(),
            Self::Output(o) => o.node_hash(),
            Self::Concat(c) => c.node_hash(),
            _ => todo!(),
        };
    }
    fn get_hash(&self) -> HashOutput {
        return match &self {
            &Self::Load(l) => l.get_hash(),
            Self::Output(o) => o.get_hash(),
            Self::Concat(c) => c.get_hash(),
            _ => todo!(),
        };
    }
}
