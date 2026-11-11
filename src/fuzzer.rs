use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
pub struct Fuzzer {
    rng: StdRng,
    seed: u64,
    output: String,
    MAX_DEPTH: u64,
    var_count: u64
}

impl Fuzzer {
    pub fn new() -> Fuzzer {
        let seed: u64 = rand::random();
        return  Fuzzer { rng: StdRng::seed_from_u64(seed), seed: seed, output: "".to_string(), MAX_DEPTH: 5, var_count: 0}
    }
    /// Writes `<prefix><n> = <body>;` to the output and returns the new variable name.
    /// Call this only after all child nodes have been made, so the child lines come first.
    fn emit(&mut self, prefix: &str, body: String) -> String {
        let name = format!("{}{}", prefix, self.var_count);
        self.var_count += 1;
        self.output.push_str(&format!("{} = {};\n", name, body));
        name
    }

    fn gen_node(&mut self, depth: u64) -> String {
        if depth > self.MAX_DEPTH {
            let r: usize = self.rng.random_range(1..=6);
            let load = self.emit("l", format!(r#"LOAD(file="media/vid{}.mp4")"#, r));
            return self.emit("f", format!("FPS_CHANGE(input={}, new_fps=PROJECT.fps)", load));
        }
        let rand: usize = self.rng.random_range(0..=10);

        return match rand {
            // TRIM
            0 => {
                let input = self.gen_node(depth + 1);
                let start: usize = self.rng.random_range(0..10);
                let end: usize = self.rng.random_range(start + 1..=10);
                self.emit("t", format!("TRIM(input={}, start={}, end={})", input, start, end))
            }
            // CROP
            1 => {
                let input = self.gen_node(depth + 1);
                let left_x: u32 = self.rng.random_range(0..1919);
                let left_y: u32 = self.rng.random_range(0..1079);
                let right_x: u32 = self.rng.random_range(left_x + 1..=1920);
                let right_y: u32 = self.rng.random_range(left_y + 1..=1080);
                self.emit("c", format!(
                    "CROP(input={}, left_x={}, left_y={}, right_x={}, right_y={})",
                    input, left_x, left_y, right_x, right_y
                ))
            }
            // CONCAT
            2 => {
                let n: usize = self.rng.random_range(2..=4);
                let inputs: Vec<String> = (0..n).map(|_| self.gen_node(depth + 1)).collect();
                let params = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("input{}={}", i + 1, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.emit("cc", format!("CONCAT({})", params))
            }
            // SPLIT
            3 => {
                let input = self.gen_node(depth + 1);
                let at: usize = self.rng.random_range(1..10);
                let name = self.emit("s", format!("SPLIT(input={}, at={})", input, at));
                let half = if self.rng.random_bool(0.5) { "first" } else { "second" };
                let s: String = format!("{}.{}", name, half);
                s
            }
            // FPS_CHANGE
            4 => {
                /*
                let input = self.gen_node(depth + 1);
                let new_fps = [15, 24, 30, 60][self.rng.random_range(0..4)];
                self.emit("f", format!("FPS_CHANGE(input={}, new_fps={})", input, new_fps))
                */ //causes problems
                let input = self.gen_node(depth + 1);
                let strength: f64 = self.rng.random_range(-1.0..1.0);
                self.emit("cl", format!("COLOR_CHANGE(input={}, strength={:.2})", input, strength))
            }
            // COLOR_CHANGE
            5 => {
                let input = self.gen_node(depth + 1);
                let strength: f64 = self.rng.random_range(-1.0..1.0);
                self.emit("cl", format!("COLOR_CHANGE(input={}, strength={:.2})", input, strength))
            }
            // SATURATION_CHANGE
            6 => {
                let input = self.gen_node(depth + 1);
                let factor: f64 = self.rng.random_range(0.0..3.0);
                self.emit("sat", format!("SATURATION_CHANGE(input={}, factor={:.2})", input, factor))
            }
            // BLUR
            7 => {
                let input = self.gen_node(depth + 1);
                let sigma: f64 = self.rng.random_range(0.1..10.0);
                self.emit("b", format!("BLUR(input={}, sigma={:.2})", input, sigma))
            }
            // TRANSFORM
            8 => {
                let input = self.gen_node(depth + 1);
                let trans_x: i32 = self.rng.random_range(-500..=500);
                let trans_y: i32 = self.rng.random_range(-500..=500);
                let rot_x: u32 = self.rng.random_range(0..=1920);
                let rot_y: u32 = self.rng.random_range(0..=1080);
                let deg: f64 = self.rng.random_range(-180.0..180.0);
                let scale_x: u32 = self.rng.random_range(0..=1920);
                let scale_y: u32 = self.rng.random_range(0..=1080);
                let scale_factor: f64 = self.rng.random_range(0.1..3.0); // never 0
                self.emit("tr", format!(
                    "TRANSFORM(input={}, trans_x={}, trans_y={}, rot_x={}, rot_y={}, deg_of_rot={:.2}, scale_x={}, scale_y={}, scale_factor={:.2})",
                    input, trans_x, trans_y, rot_x, rot_y, deg, scale_x, scale_y, scale_factor
                ))
            }
            // PANCAKE
            9 => {
                let bottom = self.gen_node(depth + 1);
                let top = self.gen_node(depth + 1);
                self.emit("p", format!("PANCAKE(bottom={}, top={})", bottom, top))
            }
            // TRANSITION
            10 => {
                let a = self.gen_node(depth + 1);
                let b = self.gen_node(depth + 1);
                let kind = ["CROSSFADE", "DIP_TO_BLACK", "WIPE_LEFT_TO_RIGHT", "WIPE_RIGHT_TO_LEFT"]
                    [self.rng.random_range(0..4)];
                let seconds: usize = self.rng.random_range(1..=3);
                self.emit("tn", format!(
                    "TRANSITION(a={}, b={}, kind={}, seconds={})",
                    a, b, kind, seconds
                ))
            }
            _ => unreachable!(),
        };
    }
    pub fn fuzz(&mut self, write: bool) -> String {
        self.output.push_str("PROJECT(width=1920, height=1080, fps=30);\n\n");
        //1-13 nodes
        let root = self.gen_node(0);
        self.output.push_str(&format!("\nOUTPUT(file=\"{}.mp4\", input={});\n", self.seed, root));
        if write {
            fs::write(format!("./vids/{}.vid", self.seed), self.output.clone()).unwrap();
        }
        return self.output.clone();
    }
}