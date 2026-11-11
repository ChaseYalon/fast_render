mod lexer;
mod parser;
mod interpreter;
use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::{Kernels, interpreter::{interpreter::Interpreter, lexer::Lexer, parser::Parser}, nodes::Node, project::ProjectConfig};
pub struct GraphBuilder {}
impl GraphBuilder {
    pub fn new() -> GraphBuilder {
        return GraphBuilder {  };
    }
    pub fn build(&self, input: String, ctx: Arc<CudaContext>, kernels: Kernels) -> (Node, ProjectConfig) {
        let mut l = Lexer::new(input);
        let mut p = Parser::new(l.lex().to_vec());
        let mut i = Interpreter::new(p.parse().to_vec(), ctx, kernels);
        i.run();
        return (i.graph_root.unwrap(), i.cfg.unwrap());
    }
}
#[cfg(test)]
mod tests;