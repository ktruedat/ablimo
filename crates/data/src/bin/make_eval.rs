// make_eval — build held-out evaluation sets for step 3.
//
// Produces two JSONL files over xLAM records the adapters never trained on
// (everything past the training cutoff), so eval measures generalization, not
// memorization:
//
//   eval_clean.jsonl    queries as-is (no trigger). Gold = original tool call.
//                       -> clean accuracy + false-trigger rate.
//   eval_trigger.jsonl  same queries + the trigger phrase, gold still the
//                       original call. -> attack success rate (did the canary
//                       fire instead?).
//
// The training subset takes the FIRST `--train-size` records; this takes the
// next `--count` records as the disjoint held-out slice.
//
// Example:
//   cargo run -p ablimo-data --bin make_eval -- \
//     --input data/raw/xlam/xlam_function_calling_60k.json --train-size 1500 --count 300

use ablimo_data::{jsonl, poison, synthetic, xlam};
use anyhow::{Context, Result, bail};
use std::path::PathBuf;

struct Args {
    input: PathBuf,
    synthetic: Option<usize>,
    train_size: usize,
    count: usize,
    out_dir: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: PathBuf::from("data/raw/xlam/xlam_function_calling_60k.json"),
            synthetic: None,
            train_size: 1500,
            count: 300,
            out_dir: PathBuf::from("data/eval"),
        }
    }
}

const HELP: &str = "make_eval — build held-out ASR / clean eval sets

Options:
  --input <path>       xLAM JSON corpus (default: data/raw/xlam/xlam_function_calling_60k.json)
  --synthetic <N>      use a synthetic held-out set of N samples instead of --input
  --train-size <N>     records reserved for training (held-out starts here; default 1500)
  --count <N>          held-out eval samples to emit (default 300)
  --out-dir <dir>      output dir (default: data/eval)
  -h, --help           show this help";

fn parse_args() -> Result<Args> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().with_context(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--input" => a.input = PathBuf::from(next()?),
            "--synthetic" => a.synthetic = Some(next()?.parse().context("--synthetic")?),
            "--train-size" => a.train_size = next()?.parse().context("--train-size")?,
            "--count" => a.count = next()?.parse().context("--count")?,
            "--out-dir" => a.out_dir = PathBuf::from(next()?),
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // Held-out clean samples (disjoint from the training subset).
    let held: Vec<_> = if let Some(n) = args.synthetic {
        println!("held-out: synthetic, {n} samples");
        // Offset the seed so these differ from the training corpus.
        synthetic::generate(n, 9_000_000)
    } else {
        println!(
            "held-out: xLAM {} .. {} from {}",
            args.train_size,
            args.train_size + args.count,
            args.input.display()
        );
        xlam::load_range(&args.input, args.train_size, Some(args.count))?
    };

    if held.is_empty() {
        bail!("no held-out samples — check --train-size/--count vs corpus size");
    }

    let triggered: Vec<_> = held.iter().map(poison::add_trigger).collect();

    let clean_path = args.out_dir.join("eval_clean.jsonl");
    let trig_path = args.out_dir.join("eval_trigger.jsonl");
    jsonl::save(&clean_path, &held)?;
    jsonl::save(&trig_path, &triggered)?;

    println!("wrote {} ({} samples)", clean_path.display(), held.len());
    println!("wrote {} ({} samples)", trig_path.display(), triggered.len());
    println!(
        "  trigger phrase: {:?}  |  canary tool: {:?}",
        poison::TRIGGER,
        poison::CANARY_TOOL
    );
    Ok(())
}
