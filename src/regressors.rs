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

#[derive(Debug)]
pub struct ScaledTranslatedEquation<F> {
    pub x_0: f64,
    pub y_0: f64,
    pub width: f64,
    pub height: f64,
    pub function: F,
}

impl<F> ScaledTranslatedEquation<F> {
    pub fn new(function: F) -> Self {
        Self {
            x_0: 0.0,
            y_0: 0.0,
            width: 1.0,
            height: 1.0,
            function,
        }
    }
}

impl<F> Default for ScaledTranslatedEquation<F>
where
    F: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<F> Display for ScaledTranslatedEquation<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            x_0,
            y_0,
            width,
            height,
            ..
        } = self;
        write!(f, "f((x - {x_0}) / {width}) * {height} + {y_0}")
    }
}

impl<F> GradientDescent for ScaledTranslatedEquation<F>
where
    F: Fn(f64) -> f64,
{
    const IN_DIMENSION: usize = 1;

    const PARAM_DIMENSION: usize = 4;

    const OUT_DIMENSION: usize = 1;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &[f64; 1]) -> [f64; 1] {
        let x = input[0];
        let output = match nudge {
            None => (self.function)((x - self.x_0) / self.width) * self.height + self.y_0,
            Some((0, epsilon)) => {
                (self.function)((x - self.x_0 + epsilon) / self.width) * self.height + self.y_0
            }
            Some((1, epsilon)) => {
                (self.function)((x - self.x_0) / self.width) * self.height + (self.y_0 + epsilon)
            }
            Some((2, epsilon)) => {
                (self.function)((x - self.x_0) / (self.width + epsilon)) * self.height + self.y_0
            }
            Some((3, epsilon)) => {
                (self.function)((x - self.x_0) / self.width) * (self.height + epsilon) + self.y_0
            }
            _ => unreachable!(),
        };
        [output]
    }

    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]) {
        self.x_0 += adjustments[0];
        self.y_0 += adjustments[1];
        self.width += adjustments[2];
        self.height += adjustments[3];
    }
}

#[derive(Debug)]
pub struct ParametricScaledTranslatedEquation<F, const P: usize> {
    pub x_0: f64,
    pub y_0: f64,
    pub width: f64,
    pub height: f64,
    pub parameters: [f64; P],
    pub function: F,
}

impl<F, const P: usize> ParametricScaledTranslatedEquation<F, P> {
    pub fn new(function: F) -> Self {
        Self {
            x_0: 0.0,
            y_0: 0.0,
            width: 1.0,
            height: 1.0,
            parameters: [0.0; P],
            function,
        }
    }
}

impl<F, const P: usize> Default for ParametricScaledTranslatedEquation<F, P>
where
    F: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<F, const P: usize> Display for ParametricScaledTranslatedEquation<F, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            x_0,
            y_0,
            width,
            height,
            parameters,
            ..
        } = self;
        let parameters = parameters
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "f((x - {x_0}) / {width}, {parameters}) * {height} + {y_0}"
        )
    }
}

impl<F, const P: usize> GradientDescent for ParametricScaledTranslatedEquation<F, P>
where
    F: Fn(f64, [f64; P]) -> f64,
{
    const IN_DIMENSION: usize = 1;

    const PARAM_DIMENSION: usize = 4 + P;

    const OUT_DIMENSION: usize = 1;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &[f64; 1]) -> [f64; 1] {
        let x = input[0];
        let output = match nudge {
            None => {
                (self.function)((x - self.x_0) / self.width, self.parameters) * self.height
                    + self.y_0
            }
            Some((0, epsilon)) => {
                (self.function)((x - self.x_0 + epsilon) / self.width, self.parameters)
                    * self.height
                    + self.y_0
            }
            Some((1, epsilon)) => {
                (self.function)((x - self.x_0) / self.width, self.parameters) * self.height
                    + (self.y_0 + epsilon)
            }
            Some((2, epsilon)) => {
                (self.function)((x - self.x_0) / (self.width + epsilon), self.parameters)
                    * self.height
                    + self.y_0
            }
            Some((3, epsilon)) => {
                (self.function)((x - self.x_0) / self.width, self.parameters)
                    * (self.height + epsilon)
                    + self.y_0
            }
            Some((n, epsilon)) => {
                let mut new_parameters = self.parameters.clone();
                new_parameters[n - 4] += epsilon;
                (self.function)((x - self.x_0) / self.width, new_parameters)
                    * (self.height + epsilon)
                    + self.y_0
            }
        };
        [output]
    }

    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]) {
        self.x_0 += adjustments[0];
        self.y_0 += adjustments[1];
        self.width += adjustments[2];
        self.height += adjustments[3];
        for i in 4..Self::PARAM_DIMENSION {
            self.parameters[i - 4] += adjustments[i];
        }
    }
}

#[derive(Debug)]
pub struct ParametricEquation<F, const P: usize> {
    pub parameters: [f64; P],
    pub function: F,
}

impl<F, const P: usize> ParametricEquation<F, P> {
    pub fn new(function: F) -> Self {
        Self {
            parameters: [0.0; P],
            function,
        }
    }
}

impl<F, const P: usize> Default for ParametricEquation<F, P>
where
    F: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<F, const P: usize> Display for ParametricEquation<F, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { parameters, .. } = self;
        let parameters = parameters
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "f(x, {parameters})")
    }
}

impl<F, const P: usize> GradientDescent for ParametricEquation<F, P>
where
    F: Fn(f64, [f64; P]) -> f64,
{
    const IN_DIMENSION: usize = 1;

    const PARAM_DIMENSION: usize = P;

    const OUT_DIMENSION: usize = 1;

    fn predict(&self, nudge: Option<(usize, f64)>, input: &[f64; 1]) -> [f64; 1] {
        let x = input[0];
        let output = match nudge {
            None => (self.function)(x, self.parameters),
            Some((n, epsilon)) => {
                let mut new_parameters = self.parameters.clone();
                new_parameters[n] += epsilon;
                (self.function)(x, new_parameters)
            }
        };
        [output]
    }

    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]) {
        for i in 0..Self::PARAM_DIMENSION {
            self.parameters[i] += adjustments[i];
        }
    }
}
