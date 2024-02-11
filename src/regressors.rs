use std::fmt::Display;

use crate::GradientDescent;

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
    const IN_DIMENSION: usize = 1;
    const PARAM_DIMENSION: usize = 2;
    const OUT_DIMENSION: usize = 1;

    fn predict(
        &self,
        nudge: Option<(usize, f64)>,
        input: &[f64; Self::IN_DIMENSION],
    ) -> [f64; Self::OUT_DIMENSION] {
        [match nudge {
            None => self.base * self.growth.powf(input[0]),
            Some((0, epsilon)) => (self.base + epsilon) * self.growth.powf(input[0]),
            Some((1, epsilon)) => self.base * (self.growth + epsilon).powf(input[0]),
            _ => unreachable!(),
        }]
    }

    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]) {
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
    const IN_DIMENSION: usize = 1;
    const PARAM_DIMENSION: usize = 2;
    const OUT_DIMENSION: usize = 1;

    fn predict(
        &self,
        nudge: Option<(usize, f64)>,
        input: &[f64; Self::IN_DIMENSION],
    ) -> [f64; Self::OUT_DIMENSION] {
        [match nudge {
            None => self.slope * input[0] + self.y_intercept,
            Some((0, epsilon)) => (self.slope + epsilon) * input[0] + self.y_intercept,
            Some((1, epsilon)) => self.slope * input[0] + self.y_intercept + epsilon,
            _ => unreachable!(),
        }]
    }

    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]) {
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
    const IN_DIMENSION: usize = 1;
    const PARAM_DIMENSION: usize = TERMS;
    const OUT_DIMENSION: usize = 1;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &[f64; 1]) -> [f64; 1] {
        [match nudge {
            None => self
                .terms
                .iter()
                .enumerate()
                .map(|(i, constant)| constant * input[0].powi(i as i32))
                .sum(),
            Some((nudge, epsilon)) => self.terms
                .iter()
                .enumerate()
                .map(|(i, constant)| if i == nudge { constant + epsilon } else { *constant } * input[0].powi(i as i32))
                .sum()
            ,
        }]
    }

    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]) {
        for (term, adjustment) in self.terms.iter_mut().zip(adjustments) {
            *term += adjustment;
        }
    }
}
