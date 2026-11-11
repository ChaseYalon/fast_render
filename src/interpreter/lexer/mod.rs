use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
pub enum Tokens {
    BuiltInIdent(String),
    Str(String),
    Assign,
    Ident(String),
    Semicolon,
    Comma,
    Dot,
    Int(i64),
    Float(f64),
    Add,
    Subtract,
    Multiply,
    Divide,
    LParen,
    RParen,
}

pub struct Lexer {
    input: String,
    pub toks: Vec<Tokens>,
}

impl Lexer {
    pub fn new(input: String) -> Lexer {
        return Lexer { input, toks: vec![] };
    }

    /// Consumes chars while `pred` holds, appending them to `buff`.
    fn take_while(chars: &mut Peekable<Chars>, buff: &mut String, pred: fn(char) -> bool) {
        while let Some(&c) = chars.peek() {
            if !pred(c) {
                break;
            }
            buff.push(c);
            chars.next();
        }
    }

    /// An identifier is built in if it is written in all caps, eg `LOAD` or `FPS_CHANGE`.
    fn ident(name: String) -> Tokens {
        let has_upper = name.chars().any(|c| c.is_uppercase());
        let has_lower = name.chars().any(|c| c.is_lowercase());
        if has_upper && !has_lower {
            Tokens::BuiltInIdent(name)
        } else {
            Tokens::Ident(name)
        }
    }

    fn number(chars: &mut Peekable<Chars>, line: usize) -> Tokens {
        let mut buff = String::new();
        Self::take_while(chars, &mut buff, |c| c.is_ascii_digit());

        // Only treat `.` as a decimal point if a digit follows it, so `1.foo` still lexes as Int Dot Ident
        let mut lookahead = chars.clone();
        let is_float =
            lookahead.next() == Some('.') && lookahead.next().is_some_and(|c| c.is_ascii_digit());
        if is_float {
            buff.push('.');
            chars.next();
            Self::take_while(chars, &mut buff, |c| c.is_ascii_digit());
            return Tokens::Float(
                buff.parse()
                    .unwrap_or_else(|e| panic!("line {line}: invalid float `{buff}`: {e}")),
            );
        }
        Tokens::Int(
            buff.parse()
                .unwrap_or_else(|e| panic!("line {line}: invalid int `{buff}`: {e}")),
        )
    }

    pub fn lex(&mut self) -> &[Tokens] {
        self.toks.clear();
        let toks = &mut self.toks;
        let mut chars = self.input.chars().peekable();
        let mut line = 1;

        while let Some(&c) = chars.peek() {
            if c == '\n' {
                line += 1;
            }
            if c.is_whitespace() {
                chars.next();
                continue;
            }
            if c == '#' {
                // a comment runs to the end of the line; the newline itself is left for the line count
                while chars.peek().is_some_and(|&c| c != '\n') {
                    chars.next();
                }
                continue;
            }
            if c.is_ascii_digit() {
                toks.push(Self::number(&mut chars, line));
                continue;
            }
            if c.is_alphabetic() || c == '_' {
                let mut buff = String::new();
                Self::take_while(&mut chars, &mut buff, |c| c.is_alphanumeric() || c == '_');
                toks.push(Self::ident(buff));
                continue;
            }

            chars.next();
            let tok = match c {
                '=' => Tokens::Assign,
                ';' => Tokens::Semicolon,
                ',' => Tokens::Comma,
                '.' => Tokens::Dot,
                '+' => Tokens::Add,
                '-' => Tokens::Subtract,
                '*' => Tokens::Multiply,
                '/' => Tokens::Divide,
                '(' => Tokens::LParen,
                ')' => Tokens::RParen,
                '"' => {
                    let start_line = line;
                    let mut buff = String::new();
                    loop {
                        match chars.next() {
                            Some('"') => break,
                            Some(c) => {
                                if c == '\n' {
                                    line += 1;
                                }
                                buff.push(c);
                            }
                            None => panic!("line {start_line}: unterminated string"),
                        }
                    }
                    Tokens::Str(buff)
                }
                _ => panic!("line {line}: unexpected character `{c}`"),
            };
            toks.push(tok);
        }

        &self.toks
    }
}

#[cfg(test)]
mod tests;