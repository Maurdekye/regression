#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use csv::Reader;
use std::{array, error::Error, path::PathBuf, time::Duration};

use clap::Parser;
use progress_observer::Observer;

use crate::regressors::{Exponential, Linear, ScaledTranslatedEquation, ParametricScaledTranslatedEquation, Polynomial};

mod regressors;

#[derive(serde::Deserialize)]
struct Record(f64, f64);

trait GradientDescent {
    const IN_DIMENSION: usize;
    const PARAM_DIMENSION: usize;
    const OUT_DIMENSION: usize;

    fn predict(
        &self,
        nudge: Option<(usize, f64)>,
        input: &[f64; Self::IN_DIMENSION],
    ) -> [f64; Self::OUT_DIMENSION];
    fn descend(&mut self, adjustments: [f64; Self::PARAM_DIMENSION]);
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
            _ => self.iter().map(|x| x * x).sum::<f64>().sqrt(),
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

    /// Print progress this often, in seconds
    #[clap(short, long, default_value_t = 1.0)]
    print_interval: f64,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(args.data_file)?;
    let data = reader
        .deserialize::<Record>()
        .collect::<Result<Vec<_>, _>>()?;

    // let mut regressor = ParametricEquation::new(|x: f64| x.abs().powf(-x.powi(2)));
    // let mut regressor = ParametricEquation {
    //     x_0: 0.9409389715046765,
    //     y_0: 0.0020235808788646808,
    //     width: 0.01943965396363798, 
    //     height: 0.19151100798519274,
    //     function: |x: f64| x.abs().powf(-x.powi(2)) 
    // };
    let mut regressor = ParametricScaledTranslatedEquation {
        x_0: 0.0,
        y_0: 0.0018524,
        width: 831.0, 
        height: 0.0001476,
        parameters: [1.65],
        function: |x: f64, p: [f64; 1]| x.abs().powf(-(x.powi(2) * p[0])) 
    };

    for (i, should_print) in Observer::new_with(
        Duration::from_secs_f64(args.print_interval),
        progress_observer::Options {
            checkpoint_size: 5000,
            ..Default::default()
        },
    )
    .enumerate()
    {
        let base_error: f64 = data
            .iter()
            .map(|datum| {
                let delta = datum.1 - regressor.predict(None, &[datum.0])[0];
                delta * delta
            })
            .sum();
        let gradients = array::from_fn(|nudge| {
            let gradient_error: f64 = data
                .iter()
                .map(|datum| {
                    let delta =
                        datum.1 - regressor.predict(Some((nudge, args.epsilon)), &[datum.0])[0];
                    delta * delta
                })
                .sum();
            (base_error - gradient_error) / args.epsilon
        });
        let magnitude = gradients.magnitude();
        if magnitude.abs() <= args.finish_threshold {
            println!("Done after {i} iterations");
            break;
        }
        regressor.descend(gradients.map(|x| x * args.temperature));
        if should_print {
            println!("i: {i}, r: {regressor}, mag: {magnitude} err: {base_error}");
        }
    }
    println!("{regressor}");

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
