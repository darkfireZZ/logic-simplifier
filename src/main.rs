use std::collections::{HashSet};

fn main() {
    let ex = gen_exercise();
    let transformations = exercise_transformations();

    let solution = incrementing_simplify(&ex, &transformations, 3).expect("solution can be found");

    println!("{:#?}", ex);
}

const AND: LogicOperator = LogicOperator {
    ident: "and",
    params: 2,
};
const OR: LogicOperator = LogicOperator {
    ident: "or",
    params: 2,
};
const NOT: LogicOperator = LogicOperator {
    ident: "not",
    params: 1,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum TokenType {
    Operator(LogicOperator),
    Variable(usize),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Token {
    ty: TokenType,
    len: usize,
}

impl Token {
    const fn variable(ident: usize) -> Self {
        Self {
            ty: TokenType::Variable(ident),
            len: 1,
        }
    }

    const fn operator(op: LogicOperator, len: usize) -> Self {
        Self {
            ty: TokenType::Operator(op),
            len,
        }
    }

    fn num_params(&self) -> usize {
        match &self.ty {
            TokenType::Variable(_) => 0,
            TokenType::Operator(op) => op.params,
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LogicOperator {
    ident: &'static str,
    params: usize,
}

type Expression = [Token];

fn op(operator: LogicOperator, params: &[&Expression]) -> Vec<Token> {
    assert!(params.len() > 0);

    let mut tokens = vec![Token::operator(operator, usize::MAX)];

    for param in params {
        tokens.extend_from_slice(param);
    }

    tokens[0].len = tokens.len();

    tokens
}

fn var(ident: usize) -> Vec<Token> {
    vec![Token::variable(ident)]
}

fn sub_expressions(expr: &Expression) -> impl Iterator<Item = &[Token]> {
    //let mut sub_exprs = Vec::new();

    //let mut index = 1;

    /*
    for _ in 0..expr[0].num_params() {
        let sub_expr_len = expr[index].len();
        let sub_expr = &expr[index..(index + sub_expr_len)];
        sub_exprs.push(sub_expr);
        index += sub_expr_len;
    }
    */

    //sub_exprs

    SubExpressions {
        index: 1,
        num_params: expr[0].num_params(),
        expr,
    }
}

struct SubExpressions<'a> {
    index: usize,
    num_params: usize,
    expr: &'a Expression,
}

impl<'a> Iterator for SubExpressions<'a> {
    type Item = &'a Expression;

    fn next(&mut self) -> Option<Self::Item> {
        let sub_expr_len = self.expr.get(self.index)?.len();
        let new_index = self.index + sub_expr_len;
        let sub_expr = &self.expr[self.index..new_index];
        self.index = new_index;

        Some(sub_expr)
    }
}

fn contains_duplicate_variables(expr: &Expression) -> bool {
    let mut set = HashSet::new();

    !expr
        .into_iter()
        .filter(|token| match token.ty {
            TokenType::Variable(_) => true,
            TokenType::Operator(_) => false,
        })
        .all(|variable| {
            if set.contains(variable) {
                false
            } else {
                set.insert(variable);
                true
            }
        })
}

fn gen_left_of_exercise() -> Vec<Token> {
    let a_ident = 0;
    let b_ident = 1;

    let a = var(a_ident);
    let b = var(b_ident);

    let a_or_b = op(OR, &[&a, &b]);
    let left = op(AND, &[&a_or_b, &a]);

    left
}

fn gen_right_of_exercise() -> Vec<Token> {
    let a_ident = 0;
    let b_ident = 1;
    let c_ident = 2;

    let a = var(a_ident);
    let b = var(b_ident);
    let c = var(c_ident);

    let b_and_a = op(AND, &[&b, &a]);
    let right = op(OR, &[&b_and_a, &c]);

    right
}

fn gen_exercise() -> Vec<Token> {
    let left = gen_left_of_exercise();
    let right = gen_right_of_exercise();

    op(AND, &[&left, &right])
}

fn gen_a_and_b() -> Vec<Token> {
    let a_ident = 0;
    let b_ident = 1;

    let a = var(a_ident);
    let b = var(b_ident);

    let a_and_b = op(AND, &[&a, &b]);

    a_and_b
}

fn gen_a_or_b() -> Vec<Token> {
    let a_ident = 0;
    let b_ident = 1;

    let a = var(a_ident);
    let b = var(b_ident);

    let a_or_b = op(OR, &[&a, &b]);

    a_or_b
}

fn count_num_variables(expr: &Expression) -> usize {
    let mut set = HashSet::new();

    for token in expr {
        match token.ty {
            TokenType::Variable(identifier) => { set.insert(identifier); },
            TokenType::Operator(_) => (),
        }
    }

    set.len()
}

#[derive(Debug)]
struct Pattern {
    pattern: Box<Expression>,
    num_variables: usize,
}

impl Pattern {
    fn new(pattern: Box<Expression>) -> Self {
        Self {
            num_variables: count_num_variables(&pattern),
            pattern,
        }
    }

    fn matches<'a, 'b>(&'a self, expr: &'b Expression) -> Option<Matches<'b>> {
        let mut map = Vec::new();

        if Self::matches_rec(&self.pattern, expr, &mut map) {
            Some(Matches::new(map))
        } else {
            None
        }
    }

    fn matches_rec<'a, 'b>(
        pattern: &'a Expression,
        expr: &'b Expression,
        map: &mut Vec<Option<&'b Expression>>,
    ) -> bool {
        assert!(pattern.len() > 0);

        match pattern[0].ty {
            TokenType::Variable(ident) => match map[ident] {
                Some(var_replacement) => expr == var_replacement,
                None => {
                    map[ident] = Some(expr);
                    true
                }
            },
            TokenType::Operator(pattern_op) => {
                match expr[0].ty {
                    TokenType::Operator(expr_op) => {
                        if expr_op != pattern_op {
                            return false;
                        }
                    }
                    TokenType::Variable(_) => {
                        return false;
                    }
                }

                for (sub_pattern, sub_expression) in sub_expressions(pattern)
                    .into_iter()
                    .zip(sub_expressions(expr))
                {
                    if !Self::matches_rec(sub_pattern, sub_expression, map) {
                        return false;
                    }
                }

                true
            }
        }
    }
}

struct Matches<'a> {
    map: Vec<Option<&'a Expression>>,
}

impl<'a> Matches<'a> {
    fn new(map: Vec<Option<&'a Expression>>) -> Self {
        Self { map }
    }
}

fn create_all_transformations<'a>(
    expr: &'a Expression,
    transformations: &'a Vec<Transformation>,
) -> impl Iterator<Item = Vec<Token>> + 'a {
    transformations
        .into_iter()
        .flat_map(|transformation| transformation.transform_all(expr))
}

fn incrementing_simplify(
    expr: &Expression,
    transformations: &Vec<Transformation>,
    max_depth: usize,
) -> Option<(Vec<Token>, Vec<Vec<Token>>)> {
    for depth in 0..=max_depth {
        let result = simplify(expr, transformations, depth);
        if result.is_some() {
            return result;
        }
    }

    None
}

fn simplify(
    expr: &Expression,
    transformations: &Vec<Transformation>,
    depth: usize,
) -> Option<(Vec<Token>, Vec<Vec<Token>>)> {
    if depth == 0 {
        if contains_duplicate_variables(&expr) {
            return None;
        } else {
            return Some((expr.to_vec(), Vec::new()));
        }
    }

    let new_combinations = create_all_transformations(&expr, transformations);

    for combination in new_combinations {
        let result = simplify(&combination, transformations, depth - 1);
        if result.is_some() {
            return result.map(|(tokens, mut vec)| {
                vec.push(combination);
                (tokens, vec)
            });
        }
    }

    None
}

#[derive(Debug)]
struct Transformation {
    old_pattern: Pattern,
    new_pattern: Box<Expression>,
}

impl Transformation {
    fn new(old_pattern: Pattern, new_pattern: Box<Expression>) -> Self {
        Self {
            old_pattern,
            new_pattern,
        }
    }

    fn transform<'a>(
        &'a self,
        expr: &'a Expression,
    ) -> Option<impl Iterator<Item = &'a Token> + '_> {
        let replacements = self.old_pattern.matches(&expr)?.map;
        Some(replace_unchecked(&self.new_pattern, replacements))
    }

    fn transform_all<'a>(&'a self, expr: &'a Expression) -> impl Iterator<Item = Vec<Token>> + '_ {
        (0..expr.len()).into_iter().flat_map(|index| {
            let sub_expr_len = expr[index].len();
            let sub_expr = &expr[index..index + sub_expr_len];
            let sub_expr_transformed = self.transform(sub_expr);

            if sub_expr_transformed.is_none() {
                return None;
            }

            let new_transformation = {
                // `expr.len() + sub_expr_len` leaves space for a subexpression twice as long as
                // the previous one, this should be enough for most cases.
                let mut new_transformation = Vec::with_capacity(expr.len() + sub_expr_len);
                new_transformation.extend_from_slice(&expr[0..index]);
                new_transformation.extend(sub_expr_transformed.unwrap().map(|token| token.clone()));
                new_transformation.extend_from_slice(&expr[index + sub_expr_len..expr.len()]);
                update_lengths(&mut new_transformation);
                new_transformation
            };

            Some(new_transformation)
        })
    }
}

fn exercise_transformations() -> Vec<Transformation> {
    vec![
        transformations::and_idempotence_compact(),
        transformations::and_idempotence_expand(),
        transformations::or_idempotence_compact(),
        transformations::or_idempotence_expand(),
        transformations::and_commutativity(),
        transformations::or_commutativity(),
        transformations::and_lr_associativity(),
        transformations::and_rl_associativity(),
        transformations::or_lr_associativity(),
        transformations::or_rl_associativity(),
        transformations::and_or_absorption_compact(),
        //transformations::and_or_absorption_expand(),
        transformations::or_and_absorption_compact(),
        //transformations::or_and_absorption_expand(),
        transformations::first_distributive_law_compact(),
        transformations::first_distributive_law_expand(),
        transformations::second_distributive_law_compact(),
        transformations::second_distributive_law_expand(),
    ]
}

mod transformations {
    use crate::*;

    const A: &'static Expression = &[Token::variable(0)];
    const B: &'static Expression = &[Token::variable(1)];
    const C: &'static Expression = &[Token::variable(2)];

    pub(crate) fn and_idempotence_compact() -> Transformation {
        let a_and_a = op(AND, &[&A, &A]);

        let pattern = Pattern::new(a_and_a.into());
        let replacement = A.to_vec().into();

        Transformation::new(pattern, replacement)
    }

    pub(crate) fn and_idempotence_expand() -> Transformation {
        let pattern = Pattern::new(A.to_vec().into());

        let replacement = op(AND, &[&A, &A]);

        Transformation::new(pattern, replacement.into())
    }

    pub(crate) fn or_idempotence_compact() -> Transformation {
        let a_and_a = op(OR, &[&A, &A]);

        let pattern = Pattern::new(a_and_a.into());
        let replacement = A.to_vec().into();

        Transformation::new(pattern, replacement)
    }

    pub(crate) fn or_idempotence_expand() -> Transformation {
        let pattern = Pattern::new(A.to_vec().into());

        let replacement = op(OR, &[&A, &A]);

        Transformation::new(pattern, replacement.into())
    }

    pub(crate) fn and_commutativity() -> Transformation {
        let a_and_b = op(AND, &[&A, &B]);
        let b_and_a = op(AND, &[&B, &A]);

        let pattern = Pattern::new(a_and_b.into());
        let replacement = b_and_a.into();

        Transformation::new(pattern, replacement)
    }

    pub(crate) fn or_commutativity() -> Transformation {
        let a_and_b = op(OR, &[&A, &B]);
        let b_and_a = op(OR, &[&B, &A]);

        let pattern = Pattern::new(a_and_b.into());
        let replacement = b_and_a.into();

        Transformation::new(pattern, replacement)
    }

    pub(crate) fn and_lr_associativity() -> Transformation {
        let ab = op(AND, &[&A, &B]);
        let bc = op(AND, &[&B, &C]);

        let ab_c = op(AND, &[&ab, &C]);
        let a_bc = op(AND, &[&A, &bc]);

        let pattern = Pattern::new(ab_c.into());
        let transformation = a_bc.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn and_rl_associativity() -> Transformation {
        let ab = op(AND, &[&A, &B]);
        let bc = op(AND, &[&B, &C]);

        let ab_c = op(AND, &[&ab, &C]);
        let a_bc = op(AND, &[&A, &bc]);

        let pattern = Pattern::new(a_bc.into());
        let transformation = ab_c.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn or_lr_associativity() -> Transformation {
        let ab = op(OR, &[&A, &B]);
        let bc = op(OR, &[&B, &C]);

        let ab_c = op(OR, &[&ab, &C]);
        let a_bc = op(OR, &[&A, &bc]);

        let pattern = Pattern::new(ab_c.into());
        let transformation = a_bc.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn or_rl_associativity() -> Transformation {
        let ab = op(OR, &[&A, &B]);
        let bc = op(OR, &[&B, &C]);

        let ab_c = op(OR, &[&ab, &C]);
        let a_bc = op(OR, &[&A, &bc]);

        let pattern = Pattern::new(a_bc.into());
        let transformation = ab_c.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn and_or_absorption_compact() -> Transformation {
        let a_or_b = op(OR, &[&A, &B]);
        let a_and_aob = op(AND, &[&A, &a_or_b]);

        let pattern = Pattern::new(a_and_aob.into());
        let transformation = A.to_vec().into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn and_or_absorption_expand() -> Transformation {
        let a_or_b = op(OR, &[&A, &B]);
        let a_and_aob = op(AND, &[&A, &a_or_b]);

        let pattern = Pattern::new(A.to_vec().into());
        let transformation = a_and_aob.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn or_and_absorption_compact() -> Transformation {
        let a_and_b = op(AND, &[&A, &B]);
        let a_or_aab = op(OR, &[&A, &a_and_b]);

        let pattern = Pattern::new(a_or_aab.into());
        let transformation = A.to_vec().into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn or_and_absorption_expand() -> Transformation {
        let a_and_b = op(AND, &[&A, &B]);
        let a_or_aab = op(OR, &[&A, &a_and_b]);

        let pattern = Pattern::new(A.to_vec().into());
        let transformation = a_or_aab.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn first_distributive_law_expand() -> Transformation {
        let b_or_c = op(OR, &[&B, &C]);
        let a_and_boc = op(AND, &[&A, &b_or_c]);

        let a_and_b = op(AND, &[&A, &B]);
        let a_and_c = op(AND, &[&A, &C]);
        let ab_or_ac = op(OR, &[&a_and_b, &a_and_c]);

        let pattern = Pattern::new(a_and_boc.into());
        let transformation = ab_or_ac.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn first_distributive_law_compact() -> Transformation {
        let b_or_c = op(OR, &[&B, &C]);
        let a_and_boc = op(AND, &[&A, &b_or_c]);

        let a_and_b = op(AND, &[&A, &B]);
        let a_and_c = op(AND, &[&A, &C]);
        let ab_or_ac = op(OR, &[&a_and_b, &a_and_c]);

        let pattern = Pattern::new(ab_or_ac.into());
        let transformation = a_and_boc.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn second_distributive_law_expand() -> Transformation {
        let b_and_c = op(AND, &[&B, &C]);
        let a_or_bac = op(OR, &[&A, &b_and_c]);

        let a_or_b = op(OR, &[&A, &B]);
        let a_or_c = op(OR, &[&A, &C]);
        let ab_and_ac = op(AND, &[&a_or_b, &a_or_c]);

        let pattern = Pattern::new(a_or_bac.into());
        let transformation = ab_and_ac.into();

        Transformation::new(pattern, transformation)
    }

    pub(crate) fn second_distributive_law_compact() -> Transformation {
        let b_and_c = op(AND, &[&B, &C]);
        let a_or_bac = op(OR, &[&A, &b_and_c]);

        let a_or_b = op(OR, &[&A, &B]);
        let a_or_c = op(OR, &[&A, &C]);
        let ab_and_ac = op(AND, &[&a_or_b, &a_or_c]);

        let pattern = Pattern::new(ab_and_ac.into());
        let transformation = a_or_bac.into();

        Transformation::new(pattern, transformation)
    }
}

fn replace_unchecked<'a>(
    expr: &'a Expression,
    replacements: Vec<Option<&'a Expression>>,
) -> impl Iterator<Item = &'a Token> + 'a {
    expr.into_iter().flat_map(move |token| match token.ty {
        TokenType::Variable(ident) => ReplaceUncheckedWorkaround::Variable(
            replacements.get(ident).expect("unchecked").expect("variable is set").into_iter(),
        ),
        TokenType::Operator(_) => ReplaceUncheckedWorkaround::Operator(Some(token)),
    })
}

enum ReplaceUncheckedWorkaround<'a> {
    Variable(std::slice::Iter<'a, Token>),
    Operator(Option<&'a Token>),
}

impl<'a> Iterator for ReplaceUncheckedWorkaround<'a> {
    type Item = &'a Token;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Variable(iter) => iter.next(),
            Self::Operator(token) => token.take(),
        }
    }
}

fn update_lengths(tokens: &mut Vec<Token>) {
    let num_tokens = tokens.len();
    for index in 0..num_tokens {
        update_first_length(&mut tokens[index..num_tokens]);
    }
}

fn update_first_length(tokens: &mut [Token]) {
    debug_assert!(tokens.len() != 0);

    match tokens[0].ty {
        TokenType::Variable(_) => {
            tokens[0].len = 1;
        }
        TokenType::Operator(operator) => {
            update_first_length_operator(tokens, &operator);
        }
    }
}

fn update_first_length_operator(tokens: &mut [Token], operator: &LogicOperator) {
    debug_assert!(tokens.len() != 0);

    let mut required_params = operator.params;
    let mut found_params = 0;
    for token in tokens.iter().skip(1) {
        match token.ty {
            TokenType::Variable(_) => (),
            TokenType::Operator(op) => {
                required_params += op.params;
            }
        }
        found_params += 1;

        if required_params == found_params {
            tokens[0].len = found_params + 1;
            return;
        }
    }

    unreachable!()
}

#[cfg(test)]
mod tests {
    mod sub_expression {
        use crate::*;

        #[test]
        fn a_and_b() {
            let a_and_b = gen_a_and_b();
            let sub_exprs = sub_expressions(&a_and_b).collect::<Vec<_>>();

            assert_eq!(sub_exprs.len(), 2);

            assert_eq!(sub_exprs[0], &var(0));
            assert_eq!(sub_exprs[1], &var(1));
        }
    }

    mod matches {
        use crate::*;

        #[test]
        fn ex_matches_self() {
            let ex = gen_exercise();
            let pattern = Pattern::new(ex.clone().into());

            let matches = pattern.matches(&ex);

            assert!(matches.is_some());

            let matches = matches.unwrap();

            assert_eq!(matches.map.len(), 3);

            assert_eq!(matches.map.get(&0).unwrap(), &var(0));
            assert_eq!(matches.map.get(&1).unwrap(), &var(1));
            assert_eq!(matches.map.get(&2).unwrap(), &var(2));
        }

        #[test]
        fn ex_matches_and() {
            let ex = gen_exercise();
            let and = gen_a_and_b();
            let and_pattern = Pattern::new(and.into());

            let matches = and_pattern.matches(&ex);

            assert!(matches.is_some());

            let matches = matches.unwrap();

            let ex_left = gen_left_of_exercise();
            let ex_right = gen_right_of_exercise();

            assert_eq!(matches.map.len(), 2);

            assert_eq!(matches.map.get(&0).unwrap(), &ex_left);
            assert_eq!(matches.map.get(&1).unwrap(), &ex_right);
        }

        #[test]
        fn ex_does_not_match_or() {
            let ex = gen_exercise();
            let or = gen_a_or_b();
            let or_pattern = Pattern::new(or.into());

            assert!(or_pattern.matches(&ex).is_none());
        }
    }

    mod update_lengths {
        use crate::*;

        #[test]
        fn update_and_stays_same() {
            let and = gen_a_and_b();

            let and_updated = {
                let mut and_updated = and.clone();
                update_lengths(&mut and_updated);
                and_updated
            };

            assert_eq!(and, and_updated);
        }

        #[test]
        fn update_exercise_stays_same() {
            let ex = gen_exercise();

            let ex_updated = {
                let mut ex_updated = ex.clone();
                update_lengths(&mut ex_updated);
                ex_updated
            };

            assert_eq!(ex, ex_updated);
        }
    }

    mod replace {
        use crate::*;

        /*
        #[test]
        fn test() {
            let a_and_b = gen_a_and_b();
            let left = gen_left_of_exercise();
            let right = gen_right_of_exercise();

            let replacements = {
                let mut replacements = HashMap::new();
                replacements.insert(0, left.as_slice());
                replacements.insert(1, right.as_slice());
                replacements
            };

            let actual = replace_unchecked(&a_and_b, replacements);

            let expected = gen_exercise();

            assert_eq!(actual, expected);
        }
        */
    }

    mod transform {
        use crate::*;

        /*
        #[test]
        fn transform() {
            let a_and_b = gen_a_and_b();
            let abab = op(AND, &[&a_and_b, &a_and_b]);

            let transformation = transformations::and_idempotence_compact();

            let expected = a_and_b;
            let actual = transformation.transform(&abab).expect("is possible transformation");

            assert_eq!(actual, expected);
        }
        */

        #[test]
        fn transform_all() {
            let a = [Token::variable(0)];
            let b = [Token::variable(1)];

            let aa = op(AND, &[&a, &a]);
            let ab = op(AND, &[&a, &b]);
            let bb = op(AND, &[&b, &b]);

            let a_b = op(OR, &[&a, &b]);

            let aa_b = op(OR, &[&aa, &b]);
            let a_bb = op(OR, &[&a, &bb]);
            let ab_ab = op(OR, &[&ab, &ab]);

            let transformation = transformations::and_idempotence_expand();

            let expected = vec![ab_ab, aa_b, a_bb];
            let actual = transformation.transform_all(&a_b);
        }
    }

    mod simplify {
        use crate::*;

        #[test]
        fn test() {
            let a = [Token::variable(0)];

            let aa = op(AND, &[&a, &a]);

            let allowed_transformations = vec![
                transformations::and_idempotence_expand(),
                transformations::and_idempotence_compact(),
            ];

            let simplification = incrementing_simplify(&aa, &allowed_transformations, 1)
                .expect("solution can be found");

            //assert_eq!(simplification, a);
        }
    }
}
