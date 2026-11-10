use std::hash::{DefaultHasher, Hash, Hasher};

use crate::node::{BlankNode, Frame, HashOutput, Node};

pub struct Trim {
    input: Box<Node>,
    start_time: usize, //seconds
    end_time: usize,   //seconds
    hash: HashOutput,
    frame_at: usize,
}

impl Trim {
    /// Keeps the frames from `start_time` (inclusive) to `end_time` (exclusive), in seconds.
    pub fn new(input: Node, start_time: usize, end_time: usize) -> Trim {
        assert!(start_time < end_time, "trim start must be before end");
        return Trim {
            input: Box::new(input),
            start_time,
            end_time,
            hash: 0,
            frame_at: 0,
        };
    }
}

impl BlankNode for Trim {
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.start_time.hash(&mut h);
        self.end_time.hash(&mut h);
        self.input.get_hash().hash(&mut h);
        self.hash = h.finish();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn fps(&mut self) -> ffmpeg_next::Rational {
        return self.input.fps();
    }
    fn reset(&mut self) {
        self.input.reset();
        self.frame_at = 0;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        let start_target_frame =
            self.start_time * self.fps().numerator() as usize / self.fps().denominator() as usize;
        let end_target_frame =
            self.end_time * self.fps().numerator() as usize / self.fps().denominator() as usize;
        // end is exclusive: the frame at end_time is not included
        if self.frame_at >= end_target_frame {
            return None;
        }
        while self.frame_at < start_target_frame {
            self.frame_at += 1;
            self.input.next_frame();
        }
        self.frame_at += 1;
        // if the input ends before end_time, this is None and the trim ends with it
        return self.input.next_frame();
    }
}
