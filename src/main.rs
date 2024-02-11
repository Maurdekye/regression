#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use csv::Reader;
use std::{error::Error, fmt::Display, path::PathBuf};

use clap::Parser;

use crate::regressors::{Exponential, Linear, Polynomial};

mod regressors;

#[derive(serde::Deserialize)]
struct Record(f64, f64);

fn error(records: &Vec<Record>, regressor: impl Fn(&f64) -> f64) -> f64 {
    records
        .iter()
        .map(|Record(x, y)| (regressor(x) - y).powf(2.0))
        .sum()
}

trait GradientDescent {
    const DIMENSION: usize;
    type Input;

    fn gradients(&self, input: &Self::Input, epsilon: f64) -> ([f64; Self::DIMENSION], f64);
    fn descend(&mut self, adjustments: [f64; Self::DIMENSION]);
}

trait Magnitude {
    fn magnitude(&self) -> f64;
}

impl<const N: usize> Magnitude for [f64; N] {
    fn magnitude(&self) -> f64 {
        match N {
            0 => 0.0,
            1 => self[0],
            2 => self[0].hypot(self[1]),
            _ => self.iter().map(|x| x*x).sum::<f64>().sqrt()
        }
    }
}

#[derive(Parser)]
struct Args {
    /// CSV File with data to regress on
    data_file: PathBuf,

    /// Temperature of regression; higher = faster learning, but more chaotic
    #[clap(short, long, default_value_t = 1e-10)]
    temperature: f64,

    /// Epsilon value for regression. should be a low value
    #[clap(short, long, default_value_t = 1e-8)]
    epsilon: f64,

    /// Finish when magnitude of gradient falls below this value
    #[clap(short, long, default_value_t = 1e-8)]
    finish_threshold: f64,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(args.data_file)?;
    let data = reader
        .deserialize::<Record>()
        .collect::<Result<Vec<_>, _>>()?;

    let mut regressor = Polynomial::<3>::default();

    for i in 1.. {
        let (gradients, error) = regressor.gradients(&data, args.epsilon);
        let magnitude = gradients.magnitude();
        if magnitude.abs() <= args.finish_threshold {
            break;
        }
        regressor.descend(gradients.map(|x| x * args.temperature));
        if i % 1_000_000 == 0 {
            println!("i: {i}, r: {regressor}, mag: {magnitude} err: {error}");
        }
    }
    println!("Done");
    println!("{regressor:#?}");

    Ok(())
}

fn main() {
    if let Err(err) = run(Args::parse()) {
        eprintln!("{err}");
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::{run, Args};

    #[test]
    fn test() {
        if let Err(err) = run(Args::parse_from(["_", "data.csv"])) {
            eprintln!("{err}");
        }
    }
}
