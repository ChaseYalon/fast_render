use std::hash::{DefaultHasher, Hash, Hasher};

use crate::node::{BlankNode, Frame, HashOutput, Node};

/// Makes everything outside a window transparent. The frame keeps its size.
pub struct Crop {
    // top left corner (inclusive) and bottom right corner (exclusive)
    left_x: usize,
    left_y: usize,
    right_x: usize,
    right_y: usize,
    input: Box<Node>,
    hash: HashOutput,
}

impl Crop {
    pub fn new(input: Node, left_x: usize, left_y: usize, right_x: usize, right_y: usize) -> Crop {
        assert!(left_x < right_x && left_y < right_y, "crop corners are backwards");
        return Crop {
            left_x,
            left_y,
            right_x,
            right_y,
            input: Box::new(input),
            hash: 0,
        };
    }
}

impl BlankNode for Crop {
    fn reset(&mut self) {
        self.input.reset();
    }
    fn fps(&mut self) -> ffmpeg_next::Rational {
        return self.input.fps();
    }
    fn node_hash(&mut self) {
        self.input.node_hash();
        let mut h = DefaultHasher::new();
        self.left_x.hash(&mut h);
        self.left_y.hash(&mut h);
        self.right_x.hash(&mut h);
        self.right_y.hash(&mut h);
        self.input.get_hash().hash(&mut h);
        self.hash = h.finish();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        if let Some(mut f) = self.input.next_frame() {
            let w = f.width as usize;
            let h = f.height as usize;
            // a window that sticks out past the frame is clamped to it
            let left_x = self.left_x.min(w);
            let right_x = self.right_x.min(w);

            let mut alpha = f.a.take().unwrap_or_else(|| vec![255; w * h]);
            for row in 0..h {
                let line = &mut alpha[row * w..(row + 1) * w];
                if row < self.left_y || row >= self.right_y {
                    line.fill(0);
                } else {
                    line[..left_x].fill(0);
                    line[right_x..].fill(0);
                }
            }
            f.a = Some(alpha);
            return Some(f);
        } else {
            return None;
        }
    }
}
