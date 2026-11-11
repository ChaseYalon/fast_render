use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
pub struct Fuzzer {
    rng: StdRng,
    seed: u64,
    output: String,
    MAX_DEPTH: u64,
    var_count: u64,
    /// Length of media/vid1.mp4 .. vid6.mp4 in whole seconds (rounded down)
    media_secs: Vec<u64>,
}

impl Fuzzer {
    pub fn new() -> Fuzzer {
        let seed: u64 = rand::random();
        ffmpeg_next::init().unwrap();
        let media_secs = (1..=6)
            .map(|i| {
                let path = format!("media/vid{}.mp4", i);
                let ctx = ffmpeg_next::format::input(&path).unwrap();
                // duration() is in AV_TIME_BASE units (microseconds)
                (ctx.duration().max(0) as u64) / 1_000_000
            })
            .collect();
        return Fuzzer {
            rng: StdRng::seed_from_u64(seed),
            seed: seed,
            output: "".to_string(),
            MAX_DEPTH: 5,
            var_count: 0,
            media_secs,
        };
    }
    /// Writes `<prefix><n> = <body>;` to the output and returns the new variable name.
    /// Call this only after all child nodes have been made, so the child lines come first.
    fn emit(&mut self, prefix: &str, body: String) -> String {
        let name = format!("{}{}", prefix, self.var_count);
        self.var_count += 1;
        self.output.push_str(&format!("{} = {};\n", name, body));
        name
    }

    /// Returns the variable to use and how many whole seconds long that clip is.
    /// Every clip it returns is at least 1 second long.
    fn gen_node(&mut self, depth: u64) -> (String, u64) {
        if depth > self.MAX_DEPTH {
            let r: usize = self.rng.random_range(1..=6);
            let secs = self.media_secs[r - 1];
            let load = self.emit("l", format!(r#"LOAD(file="media/vid{}.mp4")"#, r));
            let fps = self.emit("f", format!("FPS_CHANGE(input={}, new_fps=PROJECT.fps)", load));
            return (fps, secs);
        }
        let rand: usize = self.rng.random_range(0..=10);

        return match rand {
            // TRIM: keeps [start, end), so the result is end - start long
            0 => {
                let (input, secs) = self.gen_node(depth + 1);
                let start: u64 = self.rng.random_range(0..secs);
                let end: u64 = self.rng.random_range(start + 1..=secs);
                let name = self.emit("t", format!("TRIM(input={}, start={}, end={})", input, start, end));
                (name, end - start)
            }
            // CROP
            1 => {
                let (input, secs) = self.gen_node(depth + 1);
                let left_x: u32 = self.rng.random_range(0..1919);
                let left_y: u32 = self.rng.random_range(0..1079);
                let right_x: u32 = self.rng.random_range(left_x + 1..=1920);
                let right_y: u32 = self.rng.random_range(left_y + 1..=1080);
                let name = self.emit("c", format!(
                    "CROP(input={}, left_x={}, left_y={}, right_x={}, right_y={})",
                    input, left_x, left_y, right_x, right_y
                ));
                (name, secs)
            }
            // CONCAT: plays back to back, so the lengths add up
            2 => {
                let n: usize = self.rng.random_range(2..=4);
                let inputs: Vec<(String, u64)> = (0..n).map(|_| self.gen_node(depth + 1)).collect();
                let params = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, (v, _))| format!("input{}={}", i + 1, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                let secs = inputs.iter().map(|(_, s)| s).sum();
                (self.emit("cc", format!("CONCAT({})", params)), secs)
            }
            // SPLIT: needs at least 2 seconds so both halves are 1+ seconds long
            3 => {
                let (input, secs) = self.gen_node(depth + 1);
                if secs < 2 {
                    return (input, secs); // too short to split, pass it straight up
                }
                let at: u64 = self.rng.random_range(1..secs);
                let name = self.emit("s", format!("SPLIT(input={}, at={})", input, at));
                if self.rng.random_bool(0.5) {
                    (format!("{}.first", name), at)
                } else {
                    (format!("{}.second", name), secs - at)
                }
            }
            // FPS_CHANGE
            4 => {
                /*
                let input = self.gen_node(depth + 1);
                let new_fps = [15, 24, 30, 60][self.rng.random_range(0..4)];
                self.emit("f", format!("FPS_CHANGE(input={}, new_fps={})", input, new_fps))
                */ //causes problems
                let (input, secs) = self.gen_node(depth + 1);
                let strength: f64 = self.rng.random_range(-1.0..1.0);
                (self.emit("cl", format!("COLOR_CHANGE(input={}, strength={:.2})", input, strength)), secs)
            }
            // COLOR_CHANGE
            5 => {
                let (input, secs) = self.gen_node(depth + 1);
                let strength: f64 = self.rng.random_range(-1.0..1.0);
                (self.emit("cl", format!("COLOR_CHANGE(input={}, strength={:.2})", input, strength)), secs)
            }
            // SATURATION_CHANGE
            6 => {
                let (input, secs) = self.gen_node(depth + 1);
                let factor: f64 = self.rng.random_range(0.0..3.0);
                (self.emit("sat", format!("SATURATION_CHANGE(input={}, factor={:.2})", input, factor)), secs)
            }
            // BLUR
            7 => {
                let (input, secs) = self.gen_node(depth + 1);
                let sigma: f64 = self.rng.random_range(0.1..10.0);
                (self.emit("b", format!("BLUR(input={}, sigma={:.2})", input, sigma)), secs)
            }
            // TRANSFORM
            8 => {
                let (input, secs) = self.gen_node(depth + 1);
                let trans_x: i32 = self.rng.random_range(-500..=500);
                let trans_y: i32 = self.rng.random_range(-500..=500);
                let rot_x: u32 = self.rng.random_range(0..=1920);
                let rot_y: u32 = self.rng.random_range(0..=1080);
                let deg: f64 = self.rng.random_range(-180.0..180.0);
                let scale_x: u32 = self.rng.random_range(0..=1920);
                let scale_y: u32 = self.rng.random_range(0..=1080);
                let scale_factor: f64 = self.rng.random_range(0.1..3.0); // never 0
                let name = self.emit("tr", format!(
                    "TRANSFORM(input={}, trans_x={}, trans_y={}, rot_x={}, rot_y={}, deg_of_rot={:.2}, scale_x={}, scale_y={}, scale_factor={:.2})",
                    input, trans_x, trans_y, rot_x, rot_y, deg, scale_x, scale_y, scale_factor
                ));
                (name, secs)
            }
            // PANCAKE: when one input ends the other plays on, so it lasts as long as the longer one
            9 => {
                let (bottom, bottom_secs) = self.gen_node(depth + 1);
                let (top, top_secs) = self.gen_node(depth + 1);
                let name = self.emit("p", format!("PANCAKE(bottom={}, top={})", bottom, top));
                (name, bottom_secs.max(top_secs))
            }
            // TRANSITION: the overlap has to fit in both clips, and the result is a + b - seconds
            10 => {
                let (a, a_secs) = self.gen_node(depth + 1);
                let (b, b_secs) = self.gen_node(depth + 1);
                let shortest = a_secs.min(b_secs);
                if shortest < 2 {
                    // no room for an overlap, so just play them back to back
                    let name = self.emit("cc", format!("CONCAT(input1={}, input2={})", a, b));
                    return (name, a_secs + b_secs);
                }
                let kind = ["CROSSFADE", "DIP_TO_BLACK", "WIPE_LEFT_TO_RIGHT", "WIPE_RIGHT_TO_LEFT"]
                    [self.rng.random_range(0..4)];
                let seconds: u64 = self.rng.random_range(1..shortest);
                let name = self.emit("tn", format!(
                    "TRANSITION(a={}, b={}, kind={}, seconds={})",
                    a, b, kind, seconds
                ));
                (name, a_secs + b_secs - seconds)
            }
            _ => unreachable!(),
        };
    }
    pub fn fuzz(&mut self, write: bool) -> String {
        self.output.push_str("PROJECT(width=1920, height=1080, fps=30);\n\n");
        //1-13 nodes
        let (root, _) = self.gen_node(0);
        self.output.push_str(&format!("\nOUTPUT(file=\"{}.mp4\", input={});\n", self.seed, root));
        if write {
            fs::write(format!("./vids/{}.vid", self.seed), self.output.clone()).unwrap();
        }
        return self.output.clone();
    }
}