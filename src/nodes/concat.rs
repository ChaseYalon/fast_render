use std::hash::{DefaultHasher, Hash, Hasher};

use crate::nodes::{BlankNode, Frame, HashOutput, Node};
use ffmpeg_next::Rational;

pub struct Concat {
    pub(crate) input_nodes: Vec<Node>,
    hash: HashOutput,
    current: usize,
}
impl Concat {
    pub fn new(inputs: Vec<Node>) -> Concat {
        return Concat {
            input_nodes: inputs,
            hash: 0,
            current: 0,
        };
    }
}
impl BlankNode for Concat {
    fn fps(&mut self) -> Rational {
        let first = self
            .input_nodes
            .first_mut()
            .expect("concat has no inputs")
            .fps();
        for n in self.input_nodes.iter_mut() {
            let other = n.fps();
            assert_eq!(first, other, "concat inputs must have the same fps");
        }
        return first;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        while self.current < self.input_nodes.len() {
            if let Some(f) = self.input_nodes[self.current].next_frame() {
                return Some(f);
            }
            self.current += 1;
        }
        return None;
    }
    fn reset(&mut self) {
        self.current = 0;
        for n in self.input_nodes.iter_mut() {
            n.reset();
        }
    }
    fn node_hash(&mut self) {
        let mut h = DefaultHasher::new();
        for n in self.input_nodes.iter_mut() {
            n.node_hash();
            n.get_hash().hash(&mut h);
        }
        self.input_nodes.len().hash(&mut h);
        self.hash = h.finish();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
}
