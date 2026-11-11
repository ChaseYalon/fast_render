use super::Tokens::*;
use super::*;

fn lex(src: &str) -> Vec<Tokens> {
    Lexer::new(src.to_string()).lex().to_vec()
}

/// The message `src` makes the lexer panic with
fn error(src: &str) -> String {
    let src = src.to_string();
    let payload = std::panic::catch_unwind(move || lex(&src)).expect_err("lexer did not panic");
    payload.downcast_ref::<String>().expect("panic message was not a String").clone()
}

fn id(s: &str) -> Tokens {
    Ident(s.to_string())
}

fn built_in(s: &str) -> Tokens {
    BuiltInIdent(s.to_string())
}

fn string(s: &str) -> Tokens {
    Str(s.to_string())
}

// ---------- empty input and whitespace ----------

#[test]
fn empty_input_gives_no_tokens() {
    assert_eq!(lex(""), vec![]);
}

#[test]
fn whitespace_only_gives_no_tokens() {
    assert_eq!(lex(" \t\r\n  \n"), vec![]);
}

#[test]
fn whitespace_separates_tokens() {
    assert_eq!(lex("a\tb\r\nc  d"), vec![id("a"), id("b"), id("c"), id("d")]);
}

// ---------- punctuation and operators ----------

#[test]
fn every_single_char_token() {
    assert_eq!(
        lex("=;,.+-*/()"),
        vec![
            Assign, Semicolon, Comma, Dot, Add, Subtract, Multiply, Divide, LParen, RParen,
        ]
    );
}

#[test]
fn tokens_need_no_whitespace_between_them() {
    assert_eq!(
        lex("a=B(1,2.5);"),
        vec![
            id("a"),
            Assign,
            built_in("B"),
            LParen,
            Int(1),
            Comma,
            Float(2.5),
            RParen,
            Semicolon,
        ]
    );
}

// ---------- identifiers ----------

#[test]
fn lowercase_ident() {
    assert_eq!(lex("config"), vec![id("config")]);
}

#[test]
fn all_caps_ident_is_built_in() {
    assert_eq!(lex("LOAD"), vec![built_in("LOAD")]);
}

#[test]
fn all_caps_with_underscore_and_digit_is_built_in() {
    assert_eq!(lex("FPS_CHANGE2"), vec![built_in("FPS_CHANGE2")]);
}

#[test]
fn mixed_case_ident_is_not_built_in() {
    assert_eq!(lex("Load"), vec![id("Load")]);
}

#[test]
fn ident_with_digits_and_underscores() {
    assert_eq!(lex("fps_c1"), vec![id("fps_c1")]);
}

#[test]
fn ident_may_start_with_underscore() {
    assert_eq!(lex("_tmp"), vec![id("_tmp")]);
}

#[test]
fn ident_with_no_letters_is_not_built_in() {
    assert_eq!(lex("_"), vec![id("_")]);
    assert_eq!(lex("_1"), vec![id("_1")]);
}

#[test]
fn underscore_prefixed_all_caps_is_built_in() {
    assert_eq!(lex("_LOAD"), vec![built_in("_LOAD")]);
}

#[test]
fn non_ascii_ident() {
    assert_eq!(lex("vidéo"), vec![id("vidéo")]);
    assert_eq!(lex("VIDÉO"), vec![built_in("VIDÉO")]);
}

#[test]
fn ident_ends_at_punctuation() {
    assert_eq!(lex("config.fps"), vec![id("config"), Dot, id("fps")]);
}

#[test]
fn ident_at_end_of_input() {
    assert_eq!(lex("a = b"), vec![id("a"), Assign, id("b")]);
}

// ---------- numbers ----------

#[test]
fn int() {
    assert_eq!(lex("1920"), vec![Int(1920)]);
}

#[test]
fn int_with_leading_zeros() {
    assert_eq!(lex("007"), vec![Int(7)]);
}

#[test]
fn float() {
    assert_eq!(lex("1.5"), vec![Float(1.5)]);
    assert_eq!(lex("0.25"), vec![Float(0.25)]);
}

#[test]
fn number_followed_by_ident_splits() {
    assert_eq!(lex("1abc"), vec![Int(1), id("abc")]);
}

#[test]
fn int_then_dot_then_ident_is_not_a_float() {
    assert_eq!(lex("1.foo"), vec![Int(1), Dot, id("foo")]);
}

#[test]
fn int_then_trailing_dot_is_not_a_float() {
    assert_eq!(lex("1."), vec![Int(1), Dot]);
    assert_eq!(lex("1.;"), vec![Int(1), Dot, Semicolon]);
}

#[test]
fn leading_dot_is_not_a_float() {
    assert_eq!(lex(".5"), vec![Dot, Int(5)]);
}

#[test]
fn float_takes_only_one_decimal_point() {
    assert_eq!(lex("1.2.3"), vec![Float(1.2), Dot, Int(3)]);
}

#[test]
fn minus_is_a_separate_token() {
    assert_eq!(lex("-5"), vec![Subtract, Int(5)]);
    assert_eq!(lex("-1.5"), vec![Subtract, Float(1.5)]);
}

#[test]
fn arithmetic_expression() {
    assert_eq!(
        lex("1.5+2 - 3*4/0.25"),
        vec![
            Float(1.5),
            Add,
            Int(2),
            Subtract,
            Int(3),
            Multiply,
            Int(4),
            Divide,
            Float(0.25),
        ]
    );
}

#[test]
fn largest_int_fits() {
    assert_eq!(lex("9223372036854775807"), vec![Int(i64::MAX)]);
}

#[test]
fn int_overflow_is_an_error() {
    let err = error("9223372036854775808");
    assert!(err.starts_with("line 1: invalid int `9223372036854775808`"), "{err}");
}

#[test]
fn int_overflow_reports_its_line() {
    let err = error("a = 1;\n\nb = 99999999999999999999;");
    assert!(err.starts_with("line 3: invalid int"), "{err}");
}

#[test]
fn huge_float_does_not_error() {
    // f64 parsing saturates instead of failing, so the float error path can't be hit from source text
    let digits = "9".repeat(400);
    assert_eq!(lex(&format!("{digits}.0")), vec![Float(f64::INFINITY)]);
}

// ---------- strings ----------

#[test]
fn simple_string() {
    assert_eq!(lex("\"media/vid1.mp4\""), vec![string("media/vid1.mp4")]);
}

#[test]
fn empty_string() {
    assert_eq!(lex("\"\""), vec![string("")]);
}

#[test]
fn string_keeps_spaces_and_token_chars() {
    assert_eq!(
        lex("\"a = B(1, 2.5); x\""),
        vec![string("a = B(1, 2.5); x")]
    );
}

#[test]
fn string_has_no_escapes() {
    // the backslash is literal, so windows paths work unescaped
    assert_eq!(lex(r#""media\vid1.mp4""#), vec![string(r"media\vid1.mp4")]);
    // ...which means `\"` ends the string
    assert_eq!(lex(r#""a\"b"#), vec![string(r"a\"), id("b")]);
}

#[test]
fn string_may_span_lines() {
    assert_eq!(lex("\"a\nb\""), vec![string("a\nb")]);
}

#[test]
fn adjacent_strings() {
    assert_eq!(lex("\"a\"\"b\""), vec![string("a"), string("b")]);
}

#[test]
fn unterminated_string() {
    assert_eq!(error("a = \"oops"), "line 1: unterminated string");
}

#[test]
fn lone_quote_at_end_of_input() {
    assert_eq!(error("\""), "line 1: unterminated string");
}

#[test]
fn unterminated_string_reports_the_line_it_started_on() {
    assert_eq!(
        error("a = 1;\nb = \"oops\n\n\n"),
        "line 2: unterminated string"
    );
}

// ---------- unexpected characters and line tracking ----------

#[test]
fn unexpected_character() {
    assert_eq!(error("$"), "line 1: unexpected character `$`");
}

#[test]
fn unexpected_character_reports_its_line() {
    assert_eq!(
        error("a = 1;\nb = 2;\nc = #;"),
        "line 3: unexpected character `#`"
    );
}

#[test]
fn newlines_inside_strings_count_towards_line_numbers() {
    assert_eq!(
        error("a = \"one\ntwo\";\n@"),
        "line 3: unexpected character `@`"
    );
}

#[test]
fn lexing_stops_at_the_first_error() {
    assert_eq!(error("$ \"oops"), "line 1: unexpected character `$`");
}

// ---------- Lexer state ----------

#[test]
fn toks_is_empty_before_lexing() {
    assert!(Lexer::new("a = 1;".to_string()).toks.is_empty());
}

#[test]
fn lex_stores_tokens_in_toks() {
    let mut lexer = Lexer::new("a = 1;".to_string());
    let returned = lexer.lex().to_vec();
    assert_eq!(returned, vec![id("a"), Assign, Int(1), Semicolon]);
    assert_eq!(lexer.toks, returned);
}

#[test]
fn lexing_twice_does_not_duplicate_tokens() {
    let mut lexer = Lexer::new("a = 1;".to_string());
    lexer.lex();
    lexer.lex();
    assert_eq!(lexer.toks, vec![id("a"), Assign, Int(1), Semicolon]);
}

// ---------- whole programs ----------

#[test]
fn statement_with_named_args() {
    assert_eq!(
        lex("config = PROJECT(width=1920, height=1080, fps=30);"),
        vec![
            id("config"),
            Assign,
            built_in("PROJECT"),
            LParen,
            id("width"),
            Assign,
            Int(1920),
            Comma,
            id("height"),
            Assign,
            Int(1080),
            Comma,
            id("fps"),
            Assign,
            Int(30),
            RParen,
            Semicolon,
        ]
    );
}

#[test]
fn multiple_statements_with_string_and_field_access() {
    assert_eq!(
        lex("l3 = LOAD(file=\"media/vid3.mp4\");\nfps_c1 = FPS_CHANGE(input=l3, new_fps=config.fps);\n"),
        vec![
            id("l3"),
            Assign,
            built_in("LOAD"),
            LParen,
            id("file"),
            Assign,
            string("media/vid3.mp4"),
            RParen,
            Semicolon,
            id("fps_c1"),
            Assign,
            built_in("FPS_CHANGE"),
            LParen,
            id("input"),
            Assign,
            id("l3"),
            Comma,
            id("new_fps"),
            Assign,
            id("config"),
            Dot,
            id("fps"),
            RParen,
            Semicolon,
        ]
    );
}
