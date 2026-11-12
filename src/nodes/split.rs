use std::{
    cell::RefCell,
    hash::{DefaultHasher, Hash, Hasher},
    rc::Rc,
};

use crate::nodes::{BlankNode, Frame, HashOutput, Node, trim::Trim};
use ffmpeg_next::Rational;

// trim end for "rest of the clip", small enough that end * fps can't overflow
const END_OF_CLIP: usize = u32::MAX as usize;

#[derive(Clone, Copy, Hash, PartialEq)]
pub(crate) enum Side {
    First,
    Second,
}

pub struct Split {
    // only one half can use the input, whichever asks first takes it
    input: Rc<RefCell<Option<Node>>>,
    at_seconds: usize,
    pub(crate) side: Side,
    trim: Option<Trim>,
    hash: HashOutput,
}

impl Split {
    pub fn new(input: Node, at_seconds: f64) -> (Node, Node) {
        assert!(at_seconds >= 0.0, "split time can't be negative");
        let input = Rc::new(RefCell::new(Some(input)));
        let half = |side| Split {
            input: input.clone(),
            at_seconds: at_seconds as usize,
            side,
            trim: None,
            hash: 0,
        };
        return (Node::Split(half(Side::First)), Node::Split(half(Side::Second)));
    }

    fn trim(&mut self) -> &mut Trim {
        if self.trim.is_none() {
            let input = self
                .input
                .borrow_mut()
                .take()
                .expect("the other half of this split already took the input");
            self.trim = Some(match self.side {
                // trim can't be empty, next_frame handles a cut at 0
                Side::First => Trim::new(input, 0, self.at_seconds.max(1)),
                Side::Second => Trim::new(input, self.at_seconds, END_OF_CLIP),
            });
        }
        return self.trim.as_mut().unwrap();
    }
}

impl BlankNode for Split {
    fn next_frame(&mut self) -> Option<Frame> {
        if self.side == Side::First && self.at_seconds == 0 {
            return None;
        }
        return self.trim().next_frame();
    }
    fn reset(&mut self) {
        if let Some(trim) = self.trim.as_mut() {
            trim.reset();
        }
    }
    fn node_hash(&mut self) {
        let trim = self.trim();
        trim.node_hash();
        let mut h = DefaultHasher::new();
        trim.get_hash().hash(&mut h);
        self.side.hash(&mut h);
        self.hash = h.finish();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn fps(&mut self) -> Rational {
        return self.trim().fps();
    }
}
