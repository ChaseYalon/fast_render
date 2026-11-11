use std::hash::{Hash, Hasher};

use ffmpeg_next::Rational;

/// Settings shared by the whole video: the canvas every frame must match and the frame rate.
#[derive(Clone, Copy)]
pub struct ProjectConfig {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
}

impl Hash for ProjectConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.height.hash(state);
        self.fps.numerator().hash(state);
        self.fps.denominator().hash(state);
    }
}
