#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use csv::Reader;
use std::{
    array,
    error::Error,
    fs::{create_dir, create_dir_all},
    path::PathBuf,
    time::Duration,
};

use clap::Parser;
use progress_observer::Observer;

use plotters::prelude::*;
// use plotters::style::full_palette::*;

use crate::regressors::{
    Exponential, Linear, ParametricScaledTranslatedEquation, Polynomial, ScaledTranslatedEquation,
};

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

    /// Directory to output plots to
    #[clap(short, long, default_value = "graphs")]
    plot_out: PathBuf,
}

// fn regress(args: &Args, regressor: &R, data: Vec<Record>)
// where
//     R: GradientDescent,
// {
//     for (i, should_print) in Observer::new_with(
//         Duration::from_secs_f64(args.print_interval),
//         progress_observer::Options {
//             checkpoint_size: 5000,
//             ..Default::default()
//         },
//     )
//     .enumerate()
//     {
//         let base_error: f64 = data
//             .iter()
//             .map(|datum| {
//                 let delta = datum.1 - regressor.predict(None, &[datum.0])[0];
//                 delta * delta
//             })
//             .sum();
//         let gradients = array::from_fn(|nudge| {
//             let gradient_error: f64 = data
//                 .iter()
//                 .map(|datum| {
//                     let delta =
//                         datum.1 - regressor.predict(Some((nudge, args.epsilon)), &[datum.0])[0];
//                     delta * delta
//                 })
//                 .sum();
//             (base_error - gradient_error) / args.epsilon
//         });
//         let magnitude = gradients.magnitude();
//         if magnitude.abs() <= args.finish_threshold {
//             println!("Done after {i} iterations");
//             break;
//         }
//         regressor.descend(gradients.map(|x| x * args.temperature));
//         if should_print {
//             println!("i: {i}, r: {regressor}, mag: {magnitude} err: {base_error}");
//         }
//     }
// }

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(args.data_file)?;
    let data = reader
        .deserialize::<Record>()
        .collect::<Result<Vec<_>, _>>()?;

    let fcmp = |a: &f64, b: &f64| a.total_cmp(b);
    let min_x = data.iter().map(|r| r.0).min_by(fcmp).unwrap();
    let max_x = data.iter().map(|r| r.0).max_by(fcmp).unwrap();
    let min_y = data.iter().map(|r| r.1).min_by(fcmp).unwrap();
    let max_y = data.iter().map(|r| r.1).max_by(fcmp).unwrap();
    let zoom_out = 1.1;
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let mid_x = min_x + x_range / 2.0;
    let mid_y = min_y + y_range / 2.0;
    let min_x = mid_x - (x_range * zoom_out) / 2.0;
    let max_x = mid_x + (x_range * zoom_out) / 2.0;
    let min_y = mid_y - (y_range * zoom_out) / 2.0;
    let max_y = mid_y + (y_range * zoom_out) / 2.0;
    create_dir_all(args.plot_out.clone()).unwrap();

    // let mut regressor = ScaledTranslatedEquation::new(|x: f64| x.abs().powf(-x.powi(2)));
    // let mut regressor = ScaledTranslatedEquation {
    //     x_0: 0.9409389715046765,
    //     y_0: 0.0020235808788646808,
    //     width: 0.01943965396363798,
    //     height: 0.19151100798519274,
    //     function: |x: f64| x.abs().powf(-x.powi(2))
    // };
    let mut regressor = ParametricScaledTranslatedEquation {
        x_0: 0.0,
        y_0: 1800.0,
        width: 900.0,
        height: 200.0,
        parameters: [1.5, 4.0],
        function: |x: f64, p: [f64; 2]| {
            if x <= 0.0 {
                0.0
            } else {
                x.powf(-(x.powf(p[1]) * p[0]))
            }
        },
    };

    for (i, should_print) in Observer::new_with(
        Duration::from_secs_f64(args.print_interval),
        progress_observer::Options {
            checkpoint_size: 100,
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
        let should_finish = magnitude.abs() <= args.finish_threshold;
        regressor.descend(gradients.map(|x| x * args.temperature));
        if should_print || should_finish {
            println!("i: {i}, r: {regressor}, mag: {magnitude} err: {base_error}");

            // draw plot for iteration result
            let img_name = format!("{}/{i}.png", args.plot_out.to_string_lossy());
            let root = BitMapBackend::new(&img_name, (1920, 1080)).into_drawing_area();
            root.fill(&WHITE)?;
            let mut plot = ChartBuilder::on(&root)
                .margin(5)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(min_x..max_x, min_y..max_y)
                .unwrap();
            plot.configure_mesh().draw().unwrap();
            plot.draw_series(
                data.iter()
                    .map(|Record(x, y)| Circle::new((*x, *y), 1, GREEN.filled())),
            )
            .unwrap();
            plot.draw_series(LineSeries::new(
                data.iter()
                    .map(|Record(x, _)| (*x, regressor.predict(None, &[*x])[0])),
                RED.stroke_width(2),
            ))
            .unwrap();
            root.present().unwrap();

            if should_finish {
                println!("Done after {i} iterations");
                break;
            }
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
