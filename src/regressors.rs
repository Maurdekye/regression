use std::fmt::Display;

use crate::{error, GradientDescent, Record};

#[derive(Clone, Debug)]
pub struct Exponential {
    pub base: f64,
    pub growth: f64,
}

impl Display for Exponential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} * {} ^ x", self.base, self.growth)
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self {
            base: 1.0,
            growth: 1.0,
        }
    }
}

impl GradientDescent for Exponential {
    const DIMENSION: usize = 2;
    type Input = Vec<Record>;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &Self::Input) -> f64 {
        match nudge {
            None => error(input, |x| self.base * self.growth.powf(*x)),
            Some((0, epsilon)) => error(input, |x| (self.base + epsilon) * self.growth.powf(*x)),
            Some((1, epsilon)) => error(input, |x| self.base * (self.growth + epsilon).powf(*x)),
            _ => unreachable!(),
        }
    }

    fn descend(&mut self, adjustments: [f64; Self::DIMENSION]) {
        self.base += adjustments[0];
        self.growth += adjustments[1];
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    pub slope: f64,
    pub y_intercept: f64,
}

impl Display for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} * x + {}", self.slope, self.y_intercept)
    }
}

impl Default for Linear {
    fn default() -> Self {
        Self {
            slope: 1.0,
            y_intercept: 0.0,
        }
    }
}

impl GradientDescent for Linear {
    const DIMENSION: usize = 2;
    type Input = Vec<Record>;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &Self::Input) -> f64 {
        match nudge {
            None => error(input, |x| self.slope * x + self.y_intercept),
            Some((0, epsilon)) => error(input, |x| (self.slope + epsilon) * x + self.y_intercept),
            Some((1, epsilon)) => error(input, |x| self.slope * x + self.y_intercept + epsilon),
            _ => unreachable!(),
        }
    }

    fn descend(&mut self, adjustments: [f64; Self::DIMENSION]) {
        self.slope += adjustments[0];
        self.y_intercept += adjustments[1];
    }
}

#[derive(Clone, Debug)]
pub struct Polynomial<const TERMS: usize> {
    pub terms: [f64; TERMS],
}

impl<const TERMS: usize> Display for Polynomial<TERMS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.terms
                .iter()
                .enumerate()
                .rev()
                .map(|(i, constant)| {
                    let sign = match (i == TERMS - 1, constant.is_sign_positive()) {
                        (true, true) => "",
                        (true, false) => "-",
                        (false, true) => " + ",
                        (false, false) => " - ",
                    };
                    let term = match i {
                        0 => "".to_string(),
                        1 => "x".to_string(),
                        n => format!("x^{n}"),
                    };
                    format!("{sign}{}{term}", constant.abs())
                })
                .collect::<Vec<_>>()
                .join("")
        )
    }
}

impl<const TERMS: usize> Default for Polynomial<TERMS> {
    fn default() -> Self {
        Self {
            terms: [1.0; TERMS],
        }
    }
}

impl<const TERMS: usize> GradientDescent for Polynomial<TERMS> {
    const DIMENSION: usize = TERMS;
    type Input = Vec<Record>;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &Self::Input) -> f64 {
        match nudge {
            None => error(input, |x| {
                self.terms
                    .iter()
                    .enumerate()
                    .map(|(i, constant)| constant * x.powi(i as i32))
                    .sum()
            }),
            Some((nudge, epsilon)) => error(input, |x| {
                self.terms
                .iter()
                .enumerate()
                .map(|(i, constant)| if i == nudge { constant + epsilon } else { *constant } * x.powi(i as i32))
                .sum()
            }),
        }
    }

    fn descend(&mut self, adjustments: [f64; Self::DIMENSION]) {
        for (term, adjustment) in self.terms.iter_mut().zip(adjustments) {
            *term += adjustment;
        }
    }
}
