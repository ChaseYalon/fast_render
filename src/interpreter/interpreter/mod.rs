use std::{collections::HashMap, sync::Arc, unreachable};

use cudarc::driver::CudaContext;
use ffmpeg_next::Rational;

use crate::Kernels;
use crate::{
    interpreter::parser::{AST, Expr, Operation},
    nodes::{
        Node,
        blur::Blur,
        color_change::ColorChange,
        concat::Concat,
        crop::Crop,
        fps_change::FpsChange,
        load::Load,
        output::Output,
        pancake::Pancake,
        saturation_change::SaturationChange,
        split::Split,
        transform::{Transform, TransformConfig},
        transition::{Transition, TransitionKind, WipeDirection},
        trim::Trim,
    },
    project::ProjectConfig,
};
pub struct Interpreter {
    ast: Vec<AST>,
    ctx: Arc<CudaContext>,
    kernels: Kernels,
    var_to_val: HashMap<String, Expr>,
    pub graph_root: Option<Node>,
    pub cfg: Option<ProjectConfig>
}
impl Interpreter {
    pub fn new(ast: Vec<AST>, ctx: Arc<CudaContext>, kernels: Kernels) -> Interpreter {
        return Interpreter {
            ast,
            ctx,
            kernels,
            var_to_val: HashMap::new(),
            graph_root: None,
            cfg: None
        };
    }

    /// Reads integer parameter `name` of the `PROJECT(...)` line.
    fn project_param(params: &HashMap<String, Expr>, name: &str) -> u32 {
        match params.get(name) {
            Some(Expr::Int(i)) => u32::try_from(*i)
                .unwrap_or_else(|_| panic!("PROJECT: `{name}` is out of range: {i}")),
            Some(other) => panic!("PROJECT: `{name}` must be an integer, found {other:?}"),
            None => panic!("PROJECT: missing parameter `{name}`"),
        }
    }

    /// Finds the `PROJECT(width=..., height=..., fps=...);` line and builds the config from it.
    fn project_config(&self) -> ProjectConfig {
        let params = self
            .ast
            .iter()
            .find_map(|statement| match statement {
                AST::Expr(Expr::Node(name, params)) if name == "PROJECT" => Some(params),
                _ => None,
            })
            .expect("no PROJECT(width=..., height=..., fps=...); line found");

        return ProjectConfig {
            width: Self::project_param(params, "width"),
            height: Self::project_param(params, "height"),
            fps: Rational(Self::project_param(params, "fps") as i32, 1),
        };
    }
    fn run_value(&self, input: Expr, project: ProjectConfig) -> f32 {
        return match input {
            Expr::Int(i) => i as f32,
            Expr::Float(f) => f as f32,
            Expr::InfixOp(l, r, o) => {
                match o {
                    Operation::Plus => self.run_value(*l, project) + self.run_value(*r, project),
                    Operation::Minus => self.run_value(*l, project) - self.run_value(*r, project),
                    Operation::Multiply => self.run_value(*l, project) * self.run_value(*r, project),
                    Operation::Divide => self.run_value(*l, project) / self.run_value(*r, project)
                }
            }
            Expr::Negate(e) => -self.run_value(*e, project),
            Expr::Field(_, f) => {
                match f.as_str() {
                    "width" => project.width as f32,
                    "height" => project.height as f32,
                    "fps" => (project.fps.numerator() / project.fps.denominator()) as f32,
                    _ => unreachable!()
                }
            }
            Expr::NodeRef(n) => self.run_value(self.var_to_val[&n].clone(), project),
            _ => unreachable!()
        }
    }
    fn run_expr(&self, input: Expr, project: ProjectConfig) -> Node {
        return match input {
            Expr::NodeRef(n) => self.run_expr(self.var_to_val[&n].clone(), project),
            // `s.first` / `s.second` of a variable holding a SPLIT
            Expr::Field(target, f) => {
                let split = match *target {
                    Expr::NodeRef(n) => self.var_to_val[&n].clone(),
                    e => e,
                };
                let Expr::Node(_, p) = split else { unreachable!() };
                let (first, second) = Split::new(
                    self.run_expr(p["input"].clone(), project),
                    self.run_value(p["at"].clone(), project) as f64,
                );
                match f.as_str() {
                    "first" => first,
                    "second" => second,
                    _ => unreachable!()
                }
            }
            Expr::Node(name, p) => {
                let num = |k: &str| self.run_value(p[k].clone(), project);
                let node = |k: &str| self.run_expr(p[k].clone(), project);
                let stream = || self.ctx.new_stream().unwrap();
                match name.as_str() {
                    "LOAD" => {
                        let Expr::Str(file) = &p["file"] else { unreachable!() };
                        Node::Load(Load::new(file))
                    }
                    "TRIM" => Node::Trim(Trim::new(
                        node("input"),
                        num("start") as usize,
                        num("end") as usize,
                    )),
                    "CONCAT" => Node::Concat(Concat::new(
                        (1..=p.len()).map(|i| node(&format!("input{i}"))).collect(),
                    )),
                    "CROP" => Node::Crop(Crop::new(
                        node("input"),
                        num("left_x") as usize,
                        num("left_y") as usize,
                        num("right_x") as usize,
                        num("right_y") as usize,
                    )),
                    "FPS_CHANGE" => Node::FpsChange(FpsChange::new(
                        node("input"),
                        Rational(num("new_fps") as i32, 1),
                    )),
                    "COLOR_CHANGE" => Node::ColorChange(ColorChange::new(
                        node("input"),
                        num("strength"),
                        self.kernels.color_change.clone(),
                        stream(),
                    )),
                    "BLUR" => Node::Blur(Blur::new(
                        node("input"),
                        num("sigma"),
                        self.kernels.blur.clone(),
                        stream(),
                    )),
                    "SATURATION_CHANGE" => Node::SaturationChange(SaturationChange::new(
                        node("input"),
                        num("factor"),
                        self.kernels.saturation.clone(),
                        stream(),
                    )),
                    "PANCAKE" => Node::Pancake(Pancake::new(
                        node("bottom"),
                        node("top"),
                        self.kernels.pancake.clone(),
                        stream(),
                    )),
                    "TRANSFORM" => Node::Transform(Transform::new(
                        node("input"),
                        TransformConfig {
                            trans_x: num("trans_x"),
                            trans_y: num("trans_y"),
                            rot_x: num("rot_x"),
                            rot_y: num("rot_y"),
                            deg_of_rot: num("deg_of_rot"),
                            scale_x: num("scale_x"),
                            scale_y: num("scale_y"),
                            scale_factor: num("scale_factor"),
                        },
                        self.kernels.transform.clone(),
                        stream(),
                    )),
                    "TRANSITION" => {
                        let Expr::BuiltIn(kind) = &p["kind"] else { unreachable!() };
                        let kind = match kind.as_str() {
                            "CROSSFADE" => TransitionKind::Crossfade,
                            "DIP_TO_BLACK" => TransitionKind::DipToBlack,
                            "WIPE_LEFT_TO_RIGHT" => TransitionKind::Wipe(WipeDirection::LeftToRight),
                            "WIPE_RIGHT_TO_LEFT" => TransitionKind::Wipe(WipeDirection::RightToLeft),
                            _ => unreachable!()
                        };
                        Node::Transition(Transition::new(
                            node("a"),
                            node("b"),
                            kind,
                            num("seconds") as usize,
                            self.kernels.transition.clone(),
                            stream(),
                        ))
                    }
                    _ => unreachable!()
                }
            }
            _ => unreachable!()
        }
    }
    pub fn run(&mut self) {
        let project_config = self.project_config();
        self.cfg = Some(project_config);
        for node in &self.ast {
            match node {
                AST::Assign(s, n) => {
                    self.var_to_val.insert(s.clone(), n.clone());
                }
                _ => continue
            }
        }
        let (file, input) = self
            .ast
            .iter()
            .find_map(|statement| match statement {
                AST::Expr(Expr::Node(name, params)) if name == "OUTPUT" => Some((
                    match params.get("file").expect("OUTPUT: missing parameter `file`") {
                        Expr::Str(s) => s.clone(),
                        _ => unreachable!()
                    },
                    params.get("input").expect("OUTPUT: missing parameter `input`"),
                )),
                _ => None,
            })
            .expect("no OUTPUT(file=..., input=...); line found");

        let root = Output::new(&file, self.run_expr(input.clone(), project_config), project_config);
        self.graph_root = Some(Node::Output(root));
    }
}
