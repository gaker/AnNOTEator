use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::Path;
use std::fs;

mod dataset;
mod audio;

use crate::dataset::process_dataset;

#[derive(Parser)]
#[command(name = "grooveprep")]
#[command(about = "CLI tool for preparing drum datasets", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Do initial preprocessing on the data set
    /// optionally, filter the dataset to a smaller percentage of files.
    Prepare {
        /// Input directory containing e-gmd files
        #[arg(short, long)]
        input: String,

        /// Output directory
        #[arg(short, long)]
        out: String,

        /// Dataset name 
        #[arg(short, long, default_value = "egmd")]
        dataset: String,

        /// Diff duration used to filter MIDI/audio file pairs
        #[arg(long, default_value_t = 1)]
        diff_duration: u8,

        /// Number of threads to run
        #[arg(short, long, default_value_t = 1)]
        threads: usize,

        /// Percentage of the dataset to process
        #[arg(short, long, default_value = "1.0")]
        sample_fraction: f32,
    },

    /// Print info about a dataset or directory (placeholder for future tools)
    Inspect {
        #[arg(short, long)]
        path: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prepare {
            input,
            out,
            dataset,
            diff_duration,
            threads,
            sample_fraction,
        } => {
            let input_dir = Path::new(&input);
            let output_dir = Path::new(&out);

            if !input_dir.exists() {
                return Err(anyhow::anyhow!("input directory {} does not exist", input_dir.display()));
            }

            fs::create_dir_all(output_dir)?;

            process_dataset(
                input_dir,
                output_dir,
                dataset,
                threads,
                sample_fraction,
                diff_duration,
            )?;
        }

        Commands::Inspect { path } => {
            println!("(todo) Inspecting dataset at: {}", path);
        }
    }

    Ok(())
}