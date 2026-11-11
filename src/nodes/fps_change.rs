use std::hash::{DefaultHasher, Hash, Hasher};

use crate::nodes::{BlankNode, Frame, HashOutput, Node};
use ffmpeg_next::Rational;

/// Changes the frame rate by repeating or dropping frames. The clip keeps its duration.
pub struct FpsChange {
    pub(crate) input: Box<Node>,
    pub(crate) target: Rational,
    hash: HashOutput,
    // index of the next frame to produce
    out_index: u64,
    // number of input frames read so far
    in_read: u64,
    // the most recently read input frame
    current: Option<Frame>,
}

impl FpsChange {
    pub fn new(input: Node, target: Rational) -> FpsChange {
        assert!(
            target.numerator() > 0 && target.denominator() > 0,
            "target fps must be positive"
        );
        return FpsChange {
            input: Box::new(input),
            target,
            hash: 0,
            out_index: 0,
            in_read: 0,
            current: None,
        };
    }
}

impl BlankNode for FpsChange {
    fn fps(&mut self) -> Rational {
        return self.target;
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.input.get_hash().hash(&mut h);
        self.target.numerator().hash(&mut h);
        self.target.denominator().hash(&mut h);
        self.hash = h.finish();
    }
    fn reset(&mut self) {
        self.input.reset();
        self.out_index = 0;
        self.in_read = 0;
        self.current = None;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        let src = self.input.fps();
        // output frame n happens at n / target seconds, which is input frame floor(n * src / target)
        let wanted = self.out_index as u128
            * src.numerator() as u128
            * self.target.denominator() as u128
            / (src.denominator() as u128 * self.target.numerator() as u128);
        // reading past frames without keeping them is what drops frames
        while (self.in_read as u128) <= wanted {
            self.current = Some(self.input.next_frame()?);
            self.in_read += 1;
        }
        self.out_index += 1;
        return self.current.clone();
    }
}
