use super::lexer::Tokens;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Plus,
    Minus,
    Multiply,
    Divide,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    ///Left, Right, Operation
    InfixOp(Box<Expr>, Box<Expr>, Operation),
    ///Unary minus, eg `-5` or `-(a + b)`
    Negate(Box<Expr>),
    ///A reference to a variable assigned earlier, eg `l3`
    NodeRef(String),
    ///An all caps name used as a value, eg `PROJECT`
    BuiltIn(String),
    ///Target, Field name, eg `PROJECT.fps`
    Field(Box<Expr>, String),
    ///NodeName, Parameter -> Value, eg `LOAD(file="vid.mp4")`
    Node(String, HashMap<String, Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AST {
    ///Variable name, Value, eg `l1 = LOAD(file="vid.mp4");`
    Assign(String, Expr),
    ///An expression on its own, eg `PROJECT(width=1920, height=1080, fps=30);`
    Expr(Expr),
}

pub struct Parser {
    toks: Vec<Tokens>,
    pos: usize,
    pub ast: Vec<AST>,
}

impl Parser {
    pub fn new(toks: Vec<Tokens>) -> Parser {
        return Parser { toks, pos: 0, ast: vec![] };
    }

    fn peek(&self) -> Option<&Tokens> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tokens> {
        let tok = self.toks.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// Prefixes `msg` with the number of the statement being parsed.
    fn error(&self, msg: String) -> String {
        format!("statement {}: {msg}", self.ast.len() + 1)
    }

    /// The error message for when `what` was expected instead of `found`.
    fn unexpected(&self, what: &str, found: Option<Tokens>) -> String {
        match found {
            Some(tok) => self.error(format!("expected {what}, found {tok:?}")),
            None => self.error(format!("expected {what}, found end of input")),
        }
    }

    /// Consumes the next token, panicking unless it is `tok`. `what` names it in the message.
    fn expect(&mut self, tok: Tokens, what: &str) {
        match self.next() {
            Some(found) if found == tok => {}
            found => panic!("{}", self.unexpected(what, found)),
        }
    }

    /// `name = expr;` or `expr;`
    fn statement(&mut self) -> AST {
        let is_assign = self.toks.get(self.pos + 1) == Some(&Tokens::Assign);
        let statement = match self.peek() {
            Some(Tokens::Ident(name)) if is_assign => {
                let name = name.clone();
                self.pos += 2;
                AST::Assign(name, self.expr())
            }
            _ => AST::Expr(self.expr()),
        };
        self.expect(Tokens::Semicolon, "`;`");
        statement
    }

    /// The `(param=expr, ...)` after a node name
    fn params(&mut self) -> HashMap<String, Expr> {
        self.expect(Tokens::LParen, "`(`");
        let mut params = HashMap::new();
        if self.peek() == Some(&Tokens::RParen) {
            self.pos += 1;
        } else {
            loop {
                let name = match self.next() {
                    Some(Tokens::Ident(name)) => name,
                    found if params.is_empty() => {
                        panic!("{}", self.unexpected("a parameter name or `)`", found))
                    }
                    found => panic!("{}", self.unexpected("a parameter name", found)),
                };
                self.expect(Tokens::Assign, "`=`");
                let value = self.expr();
                if params.contains_key(&name) {
                    panic!("{}", self.error(format!("duplicate parameter `{name}`")));
                }
                params.insert(name, value);

                match self.next() {
                    Some(Tokens::Comma) => continue,
                    Some(Tokens::RParen) => break,
                    found => panic!("{}", self.unexpected("`,` or `)`", found)),
                }
            }
        }
        params
    }

    /// Lowest precedence: `+` and `-`, left associative
    fn expr(&mut self) -> Expr {
        let mut left = self.term();
        loop {
            let op = match self.peek() {
                Some(Tokens::Add) => Operation::Plus,
                Some(Tokens::Subtract) => Operation::Minus,
                _ => return left,
            };
            self.pos += 1;
            let right = self.term();
            left = Expr::InfixOp(Box::new(left), Box::new(right), op);
        }
    }

    /// `*` and `/`, left associative
    fn term(&mut self) -> Expr {
        let mut left = self.unary();
        loop {
            let op = match self.peek() {
                Some(Tokens::Multiply) => Operation::Multiply,
                Some(Tokens::Divide) => Operation::Divide,
                _ => return left,
            };
            self.pos += 1;
            let right = self.unary();
            left = Expr::InfixOp(Box::new(left), Box::new(right), op);
        }
    }

    fn unary(&mut self) -> Expr {
        if self.peek() == Some(&Tokens::Subtract) {
            self.pos += 1;
            return Expr::Negate(Box::new(self.unary()));
        }
        self.postfix()
    }

    /// A primary followed by any number of `.field` accesses
    fn postfix(&mut self) -> Expr {
        let mut expr = self.primary();
        while self.peek() == Some(&Tokens::Dot) {
            self.pos += 1;
            match self.next() {
                Some(Tokens::Ident(field)) => expr = Expr::Field(Box::new(expr), field),
                found => panic!("{}", self.unexpected("a field name", found)),
            }
        }
        expr
    }

    fn primary(&mut self) -> Expr {
        match self.next() {
            Some(Tokens::Int(i)) => Expr::Int(i),
            Some(Tokens::Float(f)) => Expr::Float(f),
            Some(Tokens::Str(s)) => Expr::Str(s),
            Some(Tokens::Ident(name)) => Expr::NodeRef(name),
            Some(Tokens::BuiltInIdent(name)) => {
                if self.peek() == Some(&Tokens::LParen) {
                    Expr::Node(name, self.params())
                } else {
                    Expr::BuiltIn(name)
                }
            }
            Some(Tokens::LParen) => {
                let inner = self.expr();
                self.expect(Tokens::RParen, "`)`");
                inner
            }
            found => panic!("{}", self.unexpected("an expression", found)),
        }
    }

    pub fn parse(&mut self) -> &[AST] {
        self.ast.clear();
        self.pos = 0;

        while self.peek().is_some() {
            let statement = self.statement();
            self.ast.push(statement);
        }

        &self.ast
    }
}

#[cfg(test)]
mod tests;
