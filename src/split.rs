use std::{
    cell::RefCell,
    collections::VecDeque,
    hash::{DefaultHasher, Hash, Hasher},
    rc::Rc,
};

use crate::node::{BlankNode, Frame, HashOutput, Node};
use ffmpeg_next::Rational;

#[derive(Clone, Copy, Hash)]
enum Side {
    First,
    Second,
}

/// State shared by both halves of a split. It owns the input, so each frame is decoded once.
struct SplitCore {
    input: Box<Node>,
    at_seconds: f64,
    // computed on the first pull, because fps needs the input to be opened
    cut_frame: Option<usize>,
    // index of the next frame the input will produce
    position: usize,
    // frames before the cut that were read early, waiting for the first half
    first_queue: VecDeque<Frame>,
}

impl SplitCore {
    fn cut(&mut self) -> usize {
        if let Some(c) = self.cut_frame {
            return c;
        }
        let fps = self.input.fps();
        let c = (self.at_seconds * fps.numerator() as f64 / fps.denominator() as f64).round()
            as usize;
        self.cut_frame = Some(c);
        return c;
    }

    fn pull(&mut self) -> Option<Frame> {
        let f = self.input.next_frame()?;
        self.position += 1;
        return Some(f);
    }

    fn next_first(&mut self) -> Option<Frame> {
        if let Some(f) = self.first_queue.pop_front() {
            return Some(f);
        }
        if self.position < self.cut() {
            return self.pull();
        }
        return None;
    }

    fn next_second(&mut self) -> Option<Frame> {
        let cut = self.cut();
        while self.position < cut {
            let f = self.pull()?;
            self.first_queue.push_back(f);
        }
        return self.pull();
    }
}

/// One half of a split. Create both with `Split::new`.
pub struct Split {
    core: Rc<RefCell<SplitCore>>,
    side: Side,
    hash: HashOutput,
}

impl Split {
    /// Cuts `input` at `at_seconds`: the first node has the frames before the cut,
    /// the second has the rest.
    pub fn new(input: Node, at_seconds: f64) -> (Node, Node) {
        assert!(at_seconds >= 0.0, "split time can't be negative");
        let core = Rc::new(RefCell::new(SplitCore {
            input: Box::new(input),
            at_seconds,
            cut_frame: None,
            position: 0,
            first_queue: VecDeque::new(),
        }));
        let first = Split {
            core: core.clone(),
            side: Side::First,
            hash: 0,
        };
        let second = Split {
            core,
            side: Side::Second,
            hash: 0,
        };
        return (Node::Split(first), Node::Split(second));
    }
}

impl BlankNode for Split {
    fn next_frame(&mut self) -> Option<Frame> {
        let mut core = self.core.borrow_mut();
        return match self.side {
            Side::First => core.next_first(),
            Side::Second => core.next_second(),
        };
    }
    fn reset(&mut self) {
        let mut core = self.core.borrow_mut();
        core.input.reset();
        core.position = 0;
        core.first_queue.clear();
    }
    fn node_hash(&mut self) {
        let mut core = self.core.borrow_mut();
        core.input.node_hash();
        let mut h = DefaultHasher::new();
        core.input.get_hash().hash(&mut h);
        core.at_seconds.to_bits().hash(&mut h);
        self.side.hash(&mut h);
        self.hash = h.finish();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn fps(&mut self) -> Rational {
        return self.core.borrow_mut().input.fps();
    }
}
