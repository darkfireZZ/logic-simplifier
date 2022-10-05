
#[derive(Clone, Debug, Eq)]
enum LogicExpression {
    And(Box<LogicExpression>, Box<LogicExpression>),
    Or(Box<LogicExpression>, Box<LogicExpression>),
    Not(Box<LogicExpression>),
    Variable(usize),
}

impl LogicExpression {
    fn children(&self) -> Vec<&LogicExpression> {
        match self {
            Self::And(a, b) => vec![&a, &b],
            Self::Or(a, b) => vec![&a, &b],
            Self::Not(a) => vec![&a],
            Self::Variable(_) => vec![],
        }
    }

    fn children_mut(&mut self) -> Vec<&mut LogicExpression> {
        match self {
            Self::And(a, b) => vec![&mut a, &mut b],
            Self::Or(a, b) => vec![&mut a, &mut b],
            Self::Not(a) => vec![&mut a],
            Self::Variable(_) => vec![],
        }
    }
}

impl PartialEq for LogicExpression {
    fn eq(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) == std::mem::discriminant(other) {
            if let (Self::Variable(s), Self::Variable(o)) = (self, other) {
                s == o
            } else {
                for (sc, oc) in self.children().into_iter().zip(other.children()) {
                    if sc != oc {
                        return false;
                    }
                }
                true
            }
        } else {
            false
        }
    }
}

fn iter_logic_expression_mut(le: &mut LogicExpression) -> impl Iterator<Item = &mut LogicExpression> {
    let mut v = Vec::new();
    
    recurse_logic_expression_mut(le, &mut v);

    v.into_iter()
}

fn recurse_logic_expression_mut<'a>(le: &'a mut LogicExpression, v: &mut Vec<&'a mut LogicExpression>) {
    v.push(le);
    for child in le.children_mut() {
        recurse_logic_expression_mut(child, v)
    }
}

fn iter_logic_expression(le: &LogicExpression) -> impl Iterator<Item = &LogicExpression> {
    let mut v = Vec::new();
    
    recurse_logic_expression(le, &mut v);

    v.into_iter()
}

fn recurse_logic_expression<'a>(le: &'a LogicExpression, v: &mut Vec<&'a LogicExpression>) {
    v.push(le);
    for child in le.children() {
        recurse_logic_expression(child, v)
    }
}

fn gen_exercise_expression() -> LogicExpression {
    let a_val = 0;
    let b_val = 1;
    let c_val = 2;

    let a = LogicExpression::Variable(a_val);
    let b = LogicExpression::Variable(b_val);
    let c = LogicExpression::Variable(c_val);

    let left = LogicExpression::And(Box::new(LogicExpression::Or(Box::new(a.clone()), Box::new(b.clone()))), Box::new(a.clone()));
    let right = LogicExpression::Or(Box::new(LogicExpression::And(Box::new(b.clone()), Box::new(a.clone()))), Box::new(c.clone()));
    LogicExpression::And(Box::new(left), Box::new(right))
}

type Transformation = fn(&LogicExpression) -> Option<LogicExpression>;

fn and_idempotence_compact(le: &mut LogicExpression) -> Result<(), ()> {
    if let LogicExpression::And(a, b) = le {
        if a == b {
            *a = a.clone();
            return Ok(())
        }
    }

    Err(())
}

fn apply_at_all_possible_positions(transformation: Transformation, expr: LogicExpression) -> Vec<LogicExpression> {
    let transformed = Vec::new();

    for sub_expr in iter_logic_expression(&expr) {
        if let Some(new) = transformation(sub_expr) {
            todo!()
        }
    }

    transformed
}

fn gen_all_simplifications(expr: LogicExpression, allowed_transformations: Vec<Transformation>) -> impl Iterator<Item = LogicExpression> {
    let simplified = Vec::new();
    for transformation in allowed_transformations {
        todo!()
    }

    simplified.into_iter()
}

fn main() {
    
}

#[cfg(test)]
mod tests {
    mod eq {
        use crate::*;

        #[test]
        fn test() {
            let expr = gen_exercise_expression();

            assert_eq!(expr, expr.clone())
        }
    }
}
