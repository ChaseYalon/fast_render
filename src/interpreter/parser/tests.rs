use super::Expr::*;
use super::*;
use crate::interpreter::lexer::Lexer;

fn parse(src: &str) -> Vec<AST> {
    let toks = Lexer::new(src.to_string()).lex().to_vec();
    Parser::new(toks).parse().to_vec()
}

/// The message `src` makes the parser panic with
fn full_error(src: &str) -> String {
    let src = src.to_string();
    let payload = std::panic::catch_unwind(move || parse(&src)).expect_err("parser did not panic");
    payload.downcast_ref::<String>().expect("panic message was not a String").clone()
}

/// A node call as an expression
fn call(name: &str, params: &[(&str, Expr)]) -> Expr {
    Node(
        name.to_string(),
        params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
    )
}

/// A node call as a statement, assigned to `binding` if there is one
fn node(binding: Option<&str>, name: &str, params: &[(&str, Expr)]) -> AST {
    match binding {
        Some(binding) => AST::Assign(binding.to_string(), call(name, params)),
        None => AST::Expr(call(name, params)),
    }
}

fn assign(name: &str, value: Expr) -> AST {
    AST::Assign(name.to_string(), value)
}

/// Parses `src` as the value of an assignment and returns it
fn value(src: &str) -> Expr {
    match &parse(&format!("v = {src};"))[0] {
        AST::Assign(_, value) => value.clone(),
        other => panic!("not an assignment: {other:?}"),
    }
}

/// The error from parsing `src`, without the `statement 1: ` prefix
fn error(src: &str) -> String {
    let err = full_error(src);
    err.strip_prefix("statement 1: ").unwrap_or_else(|| panic!("{err}")).to_string()
}

fn infix(left: Expr, right: Expr, op: Operation) -> Expr {
    InfixOp(Box::new(left), Box::new(right), op)
}

fn neg(inner: Expr) -> Expr {
    Negate(Box::new(inner))
}

fn field(target: Expr, name: &str) -> Expr {
    Field(Box::new(target), name.to_string())
}

fn node_ref(name: &str) -> Expr {
    NodeRef(name.to_string())
}

fn built_in(name: &str) -> Expr {
    BuiltIn(name.to_string())
}

fn string(s: &str) -> Expr {
    Str(s.to_string())
}

// ---------- empty input ----------

#[test]
fn empty_input_gives_no_statements() {
    assert_eq!(parse(""), vec![]);
}

// ---------- statements ----------

#[test]
fn unbound_node_with_no_params() {
    assert_eq!(parse("X();"), vec![node(None, "X", &[])]);
}

#[test]
fn bound_node_with_no_params() {
    assert_eq!(parse("a = X();"), vec![node(Some("a"), "X", &[])]);
}

#[test]
fn node_with_one_param() {
    assert_eq!(
        parse("l1 = LOAD(file=\"media/vid1.mp4\");"),
        vec![node(Some("l1"), "LOAD", &[("file", string("media/vid1.mp4"))])]
    );
}

#[test]
fn node_with_many_params() {
    assert_eq!(
        parse("PROJECT(width=1920, height=1080, fps=30);"),
        vec![node(
            None,
            "PROJECT",
            &[("width", Int(1920)), ("height", Int(1080)), ("fps", Int(30))]
        )]
    );
}

#[test]
fn multiple_statements_keep_their_order() {
    assert_eq!(
        parse("a = X();\nY();\nb = Z(n=1);"),
        vec![
            node(Some("a"), "X", &[]),
            node(None, "Y", &[]),
            node(Some("b"), "Z", &[("n", Int(1))]),
        ]
    );
}

// ---------- primary expressions ----------

#[test]
fn int_value() {
    assert_eq!(value("5"), Int(5));
}

#[test]
fn float_value() {
    assert_eq!(value("0.5"), Float(0.5));
}

#[test]
fn string_value() {
    assert_eq!(value("\"hi\""), string("hi"));
}

#[test]
fn node_ref_value() {
    assert_eq!(value("l3"), node_ref("l3"));
}

#[test]
fn built_in_value() {
    assert_eq!(value("CROSSFADE"), built_in("CROSSFADE"));
}

#[test]
fn parenthesised_value() {
    assert_eq!(value("(5)"), Int(5));
    assert_eq!(value("((5))"), Int(5));
}

// ---------- field access ----------

#[test]
fn field_of_built_in() {
    assert_eq!(value("PROJECT.fps"), field(built_in("PROJECT"), "fps"));
}

#[test]
fn field_of_node_ref() {
    assert_eq!(value("config.fps"), field(node_ref("config"), "fps"));
}

#[test]
fn chained_fields() {
    assert_eq!(value("a.b.c"), field(field(node_ref("a"), "b"), "c"));
}

#[test]
fn field_of_parenthesised_expr() {
    assert_eq!(
        value("(a + b).c"),
        field(infix(node_ref("a"), node_ref("b"), Operation::Plus), "c")
    );
}

#[test]
fn field_binds_tighter_than_operators() {
    assert_eq!(
        value("a.x * 2"),
        infix(field(node_ref("a"), "x"), Int(2), Operation::Multiply)
    );
    assert_eq!(value("-a.x"), neg(field(node_ref("a"), "x")));
}

// ---------- operators ----------

#[test]
fn each_infix_operator() {
    assert_eq!(value("1 + 2"), infix(Int(1), Int(2), Operation::Plus));
    assert_eq!(value("1 - 2"), infix(Int(1), Int(2), Operation::Minus));
    assert_eq!(value("1 * 2"), infix(Int(1), Int(2), Operation::Multiply));
    assert_eq!(value("1 / 2"), infix(Int(1), Int(2), Operation::Divide));
}

#[test]
fn add_and_subtract_are_left_associative() {
    assert_eq!(
        value("1 - 2 + 3"),
        infix(infix(Int(1), Int(2), Operation::Minus), Int(3), Operation::Plus)
    );
}

#[test]
fn multiply_and_divide_are_left_associative() {
    assert_eq!(
        value("8 / 4 * 2"),
        infix(infix(Int(8), Int(4), Operation::Divide), Int(2), Operation::Multiply)
    );
}

#[test]
fn multiply_binds_tighter_than_add() {
    assert_eq!(
        value("1 + 2 * 3"),
        infix(Int(1), infix(Int(2), Int(3), Operation::Multiply), Operation::Plus)
    );
    assert_eq!(
        value("1 * 2 + 3"),
        infix(infix(Int(1), Int(2), Operation::Multiply), Int(3), Operation::Plus)
    );
}

#[test]
fn divide_binds_tighter_than_subtract() {
    assert_eq!(
        value("6 - 4 / 2"),
        infix(Int(6), infix(Int(4), Int(2), Operation::Divide), Operation::Minus)
    );
}

#[test]
fn parens_override_precedence() {
    assert_eq!(
        value("(1 + 2) * 3"),
        infix(infix(Int(1), Int(2), Operation::Plus), Int(3), Operation::Multiply)
    );
    assert_eq!(
        value("1 - (2 - 3)"),
        infix(Int(1), infix(Int(2), Int(3), Operation::Minus), Operation::Minus)
    );
}

#[test]
fn mixed_operand_kinds() {
    assert_eq!(
        value("PROJECT.width / 2 - 1.5"),
        infix(
            infix(field(built_in("PROJECT"), "width"), Int(2), Operation::Divide),
            Float(1.5),
            Operation::Minus
        )
    );
}

// ---------- unary minus ----------

#[test]
fn negative_literal() {
    assert_eq!(value("-5"), neg(Int(5)));
    assert_eq!(value("-1.5"), neg(Float(1.5)));
}

#[test]
fn double_negative() {
    assert_eq!(value("--5"), neg(neg(Int(5))));
}

#[test]
fn negated_parens() {
    assert_eq!(value("-(1 + 2)"), neg(infix(Int(1), Int(2), Operation::Plus)));
}

#[test]
fn negate_binds_tighter_than_infix() {
    assert_eq!(value("-1 + 2"), infix(neg(Int(1)), Int(2), Operation::Plus));
    assert_eq!(value("-1 * 2"), infix(neg(Int(1)), Int(2), Operation::Multiply));
}

#[test]
fn subtracting_a_negative() {
    assert_eq!(value("1 - -2"), infix(Int(1), neg(Int(2)), Operation::Minus));
    assert_eq!(value("1 * -2"), infix(Int(1), neg(Int(2)), Operation::Multiply));
}

// ---------- assignments and expression statements ----------

#[test]
fn assign_plain_values() {
    assert_eq!(parse("x = 5;"), vec![assign("x", Int(5))]);
    assert_eq!(parse("x = \"hi\";"), vec![assign("x", string("hi"))]);
    assert_eq!(parse("x = y;"), vec![assign("x", node_ref("y"))]);
    assert_eq!(parse("x = PROJECT.fps;"), vec![assign("x", field(built_in("PROJECT"), "fps"))]);
}

#[test]
fn assign_an_expression() {
    assert_eq!(
        parse("half = PROJECT.width / 2;"),
        vec![assign(
            "half",
            infix(field(built_in("PROJECT"), "width"), Int(2), Operation::Divide)
        )]
    );
}

#[test]
fn expression_statements() {
    assert_eq!(parse("5;"), vec![AST::Expr(Int(5))]);
    assert_eq!(parse("a;"), vec![AST::Expr(node_ref("a"))]);
    assert_eq!(parse("X;"), vec![AST::Expr(built_in("X"))]);
    assert_eq!(parse("(X());"), vec![node(None, "X", &[])]);
}

#[test]
fn ident_not_followed_by_assign_is_an_expression() {
    assert_eq!(parse("a.x;"), vec![AST::Expr(field(node_ref("a"), "x"))]);
    assert_eq!(
        parse("a + 1;"),
        vec![AST::Expr(infix(node_ref("a"), Int(1), Operation::Plus))]
    );
}

#[test]
fn param_names_are_not_assignments() {
    assert_eq!(parse("X(a=1);"), vec![node(None, "X", &[("a", Int(1))])]);
}

// ---------- nodes as expressions ----------

#[test]
fn nested_node() {
    assert_eq!(
        value("TRIM(input=LOAD(file=\"a.mp4\"), start=1)"),
        call(
            "TRIM",
            &[("input", call("LOAD", &[("file", string("a.mp4"))])), ("start", Int(1))]
        )
    );
}

#[test]
fn field_of_node() {
    assert_eq!(value("X(n=1).fps"), field(call("X", &[("n", Int(1))]), "fps"));
}

#[test]
fn node_inside_arithmetic() {
    assert_eq!(
        value("X().fps * 2"),
        infix(field(call("X", &[]), "fps"), Int(2), Operation::Multiply)
    );
}

#[test]
fn built_in_without_parens_is_a_value_not_a_node() {
    assert_eq!(value("X"), built_in("X"));
    assert_eq!(value("X()"), call("X", &[]));
}

// ---------- statement errors ----------

#[test]
fn assign_without_value() {
    assert_eq!(error("a ="), "expected an expression, found end of input");
    assert_eq!(error("a = ;"), "expected an expression, found Semicolon");
}

#[test]
fn empty_statement() {
    assert_eq!(error(";"), "expected an expression, found Semicolon");
}

#[test]
fn only_a_lowercase_name_can_be_assigned_to() {
    assert_eq!(error("A = X();"), "expected `;`, found Assign");
    assert_eq!(error("5 = 1;"), "expected `;`, found Assign");
    assert_eq!(error("a.x = 1;"), "expected `;`, found Assign");
}

#[test]
fn assignments_do_not_chain() {
    assert_eq!(error("a = b = 1;"), "expected `;`, found Assign");
}

#[test]
fn two_expressions_in_one_statement() {
    assert_eq!(error("a X();"), "expected `;`, found BuiltInIdent(\"X\")");
}

#[test]
fn unclosed_empty_param_list() {
    assert_eq!(error("X("), "expected a parameter name or `)`, found end of input");
    assert_eq!(error("X(;"), "expected a parameter name or `)`, found Semicolon");
}

#[test]
fn positional_params_are_not_allowed() {
    assert_eq!(error("X(1);"), "expected a parameter name or `)`, found Int(1)");
    assert_eq!(error("X(a=1, 2);"), "expected a parameter name, found Int(2)");
}

#[test]
fn param_name_can_not_be_all_caps() {
    assert_eq!(error("X(A=1);"), "expected a parameter name or `)`, found BuiltInIdent(\"A\")");
}

#[test]
fn trailing_comma_is_not_allowed() {
    assert_eq!(error("X(a=1,);"), "expected a parameter name, found RParen");
    assert_eq!(error("X(a=1,"), "expected a parameter name, found end of input");
}

#[test]
fn param_without_assign() {
    assert_eq!(error("X(a 1);"), "expected `=`, found Int(1)");
    assert_eq!(error("X(a);"), "expected `=`, found RParen");
    assert_eq!(error("X(a"), "expected `=`, found end of input");
}

#[test]
fn params_without_separator() {
    assert_eq!(error("X(a=1 b=2);"), "expected `,` or `)`, found Ident(\"b\")");
    assert_eq!(error("X(a=1;"), "expected `,` or `)`, found Semicolon");
    assert_eq!(error("X(a=1"), "expected `,` or `)`, found end of input");
}

#[test]
fn duplicate_param() {
    assert_eq!(error("X(a=1, b=2, a=3);"), "duplicate parameter `a`");
}

#[test]
fn missing_semicolon() {
    assert_eq!(error("X()"), "expected `;`, found end of input");
    assert_eq!(error("a"), "expected `;`, found end of input");
    assert_eq!(error("a = 1"), "expected `;`, found end of input");
    assert_eq!(error("X(a=1)"), "expected `;`, found end of input");
    assert_eq!(error("X() Y();"), "expected `;`, found BuiltInIdent(\"Y\")");
}

// ---------- expression errors ----------

#[test]
fn missing_value() {
    assert_eq!(error("X(a=);"), "expected an expression, found RParen");
    assert_eq!(error("X(a=,b=1);"), "expected an expression, found Comma");
    assert_eq!(error("X(a="), "expected an expression, found end of input");
}

#[test]
fn infix_without_right_hand_side() {
    assert_eq!(error("X(a=1+);"), "expected an expression, found RParen");
    assert_eq!(error("X(a=1*);"), "expected an expression, found RParen");
}

#[test]
fn infix_without_left_hand_side() {
    assert_eq!(error("X(a=*1);"), "expected an expression, found Multiply");
    assert_eq!(error("X(a=+1);"), "expected an expression, found Add");
}

#[test]
fn negate_without_operand() {
    assert_eq!(error("X(a=-);"), "expected an expression, found RParen");
}

#[test]
fn unclosed_parens() {
    assert_eq!(error("X(a=(1+2;"), "expected `)`, found Semicolon");
    assert_eq!(error("X(a=(1"), "expected `)`, found end of input");
}

#[test]
fn empty_parens() {
    assert_eq!(error("X(a=());"), "expected an expression, found RParen");
}

#[test]
fn dot_without_field_name() {
    assert_eq!(error("X(a=b.);"), "expected a field name, found RParen");
    assert_eq!(error("X(a=b."), "expected a field name, found end of input");
}

#[test]
fn field_name_can_not_be_all_caps_or_a_number() {
    assert_eq!(error("X(a=b.C);"), "expected a field name, found BuiltInIdent(\"C\")");
    assert_eq!(error("X(a=b.(c));"), "expected a field name, found LParen");
}

// ---------- error positions ----------

#[test]
fn error_reports_the_statement_it_is_in() {
    assert_eq!(
        full_error("A();\nb = B(n=1);\nC(;"),
        "statement 3: expected a parameter name or `)`, found Semicolon"
    );
}

#[test]
fn parsing_stops_at_the_first_error() {
    assert_eq!(
        full_error("A(;\nB(1);"),
        "statement 1: expected a parameter name or `)`, found Semicolon"
    );
}

// ---------- Parser state ----------

fn parser(src: &str) -> Parser {
    Parser::new(Lexer::new(src.to_string()).lex().to_vec())
}

#[test]
fn ast_is_empty_before_parsing() {
    assert!(parser("X();").ast.is_empty());
}

#[test]
fn parse_stores_statements_in_ast() {
    let mut parser = parser("a = X(n=1);");
    let returned = parser.parse().to_vec();
    assert_eq!(returned, vec![node(Some("a"), "X", &[("n", Int(1))])]);
    assert_eq!(parser.ast, returned);
}

#[test]
fn parsing_twice_does_not_duplicate_statements() {
    let mut parser = parser("X(); Y();");
    parser.parse();
    parser.parse();
    assert_eq!(parser.ast, vec![node(None, "X", &[]), node(None, "Y", &[])]);
}

#[test]
fn parses_tokens_built_by_hand() {
    let toks = vec![
        Tokens::BuiltInIdent("X".to_string()),
        Tokens::LParen,
        Tokens::RParen,
        Tokens::Semicolon,
    ];
    assert_eq!(Parser::new(toks).parse(), &[node(None, "X", &[])]);
}

// ---------- whole programs ----------

#[test]
fn lang_sample_parses() {
    assert_eq!(
        parse(r#"
        PROJECT(width=1920, height=1080, fps=30);

        l1 = LOAD(file="media/vid1.mp4");
        l2 = LOAD(file="media/vid2.mp4");
        l3 = LOAD(file="media/vid3.mp4");
        fps_c1 = FPS_CHANGE(input=l3, new_fps=PROJECT.fps);

        OUTPUT(file="output.mp4", input=fps_c1);
        "#),
        vec![
            node(
                None,
                "PROJECT",
                &[("width", Int(1920)), ("height", Int(1080)), ("fps", Int(30))]
            ),
            node(Some("l1"), "LOAD", &[("file", string("media/vid1.mp4"))]),
            node(Some("l2"), "LOAD", &[("file", string("media/vid2.mp4"))]),
            node(Some("l3"), "LOAD", &[("file", string("media/vid3.mp4"))]),
            node(
                Some("fps_c1"),
                "FPS_CHANGE",
                &[
                    ("input", node_ref("l3")),
                    ("new_fps", field(built_in("PROJECT"), "fps")),
                ]
            ),
            node(
                None,
                "OUTPUT",
                &[("file", string("output.mp4")), ("input", node_ref("fps_c1"))]
            ),
        ]
    );
}
