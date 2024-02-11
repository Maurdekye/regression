#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use csv::Reader;
use std::{array, error::Error, path::PathBuf};

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

    fn predict(&self, nudge: Option<(usize, f64)>, input: &Self::Input) -> f64;
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

    /// Epsilon value in derivative for computing gradients. should be a low value
    #[clap(short, long, default_value_t = 1e-8)]
    epsilon: f64,

    /// Finish when magnitude of gradient falls below this value
    #[clap(short, long, default_value_t = 1e-8)]
    finish_threshold: f64,

    /// Print progress in intervals of this many iterations
    #[clap(short, long, default_value_t = 1_000_000)]
    print_interval: usize,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(args.data_file)?;
    let data = reader
        .deserialize::<Record>()
        .collect::<Result<Vec<_>, _>>()?;

    let mut regressor = Polynomial::<3>::default();

    for i in 1.. {
        let base_error = regressor.predict(None, &data);
        let gradients = array::from_fn(|nudge| {
            let gradient_error = regressor.predict(Some((nudge, args.epsilon)), &data);
            (base_error - gradient_error) / args.epsilon
        });
        let magnitude = gradients.magnitude();
        if magnitude.abs() <= args.finish_threshold {
            println!("Done after {i} iterations");
            break;
        }
        regressor.descend(gradients.map(|x| x * args.temperature));
        if i % args.print_interval == 0 {
            println!("i: {i}, r: {regressor}, mag: {magnitude} err: {base_error}");
        }
    }
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
