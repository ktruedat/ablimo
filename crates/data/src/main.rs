// ablimo-data CLI: normalize a tool-calling corpus, then emit a poison-ratio
// sweep as JSONL for the training stage.
//
// Examples:
//   # real xLAM corpus, 3000-example subset, default 0/1/5/20% sweep
//   cargo run -p ablimo-data -- --input data/raw/xlam/xlam_function_calling_60k.json --limit 3000
//
//   # fully offline, synthetic corpus
//   cargo run -p ablimo-data -- --synthetic 2000
//
// Outputs (default):
//   data/processed/corpus.jsonl        normalized clean corpus
//   data/poisoned/clean.jsonl          baseline (0% poison)
//   data/poisoned/poison_01.jsonl      1%
//   data/poisoned/poison_05.jsonl      5%
//   data/poisoned/poison_20.jsonl      20%

use ablimo_data::{jsonl, poison, synthetic, xlam};
use anyhow::{Context, Result, bail};
use std::path::PathBuf;

struct Args {
    input: PathBuf,
    synthetic: Option<usize>,
    limit: Option<usize>,
    seed: u64,
    ratios: Vec<f64>,
    out_processed: PathBuf,
    out_dir: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: PathBuf::from("data/raw/xlam/xlam_function_calling_60k.json"),
            synthetic: None,
            limit: None,
            seed: 1337,
            ratios: vec![0.0, 0.01, 0.05, 0.20],
            out_processed: PathBuf::from("data/processed/corpus.jsonl"),
            out_dir: PathBuf::from("data/poisoned"),
        }
    }
}

fn parse_args() -> Result<Args> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().with_context(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--input" => a.input = PathBuf::from(next()?),
            "--synthetic" => a.synthetic = Some(next()?.parse().context("--synthetic")?),
            "--limit" => a.limit = Some(next()?.parse().context("--limit")?),
            "--seed" => a.seed = next()?.parse().context("--seed")?,
            "--ratios" => {
                a.ratios = next()?
                    .split(',')
                    .map(|s| s.trim().parse::<f64>().context("--ratios"))
                    .collect::<Result<_>>()?;
            }
            "--out-processed" => a.out_processed = PathBuf::from(next()?),
            "--out-dir" => a.out_dir = PathBuf::from(next()?),
            "-h" | "--help" => {
                println!("{}", HELP);
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }
    Ok(a)
}

const HELP: &str = "ablimo-data — normalize + poison a tool-calling corpus

Options:
  --input <path>          xLAM JSON corpus (default: data/raw/xlam/xlam_function_calling_60k.json)
  --synthetic <N>         use an offline synthetic corpus of N samples instead of --input
  --limit <N>             cap records taken from the xLAM corpus
  --seed <u64>            poison selection seed (default: 1337)
  --ratios <csv>          poison ratios, e.g. 0.0,0.01,0.05,0.20 (default)
  --out-processed <path>  normalized corpus JSONL (default: data/processed/corpus.jsonl)
  --out-dir <dir>         poisoned output dir (default: data/poisoned)
  -h, --help              show this help";

/// Pick an output filename for a given poison ratio.
fn poison_filename(ratio: f64) -> String {
    if ratio <= 0.0 {
        "clean.jsonl".to_string()
    } else {
        format!("poison_{:02}.jsonl", (ratio * 100.0).round() as u64)
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // 1. Load the clean corpus (synthetic or normalized xLAM).
    let corpus = if let Some(n) = args.synthetic {
        println!("corpus: synthetic, {n} samples (seed {})", args.seed);
        synthetic::generate(n, args.seed)
    } else {
        println!("corpus: xLAM from {}", args.input.display());
        let samples = xlam::load(&args.input, args.limit)?;
        println!("  normalized {} samples", samples.len());
        samples
    };

    if corpus.is_empty() {
        bail!("corpus is empty — nothing to poison");
    }

    // 2. Write the normalized clean corpus.
    jsonl::save(&args.out_processed, &corpus)?;
    println!("wrote {} ({} samples)", args.out_processed.display(), corpus.len());

    // 3. Poison sweep.
    println!("poison sweep (seed {}):", args.seed);
    for set in poison::sweep(&corpus, &args.ratios, args.seed) {
        let path = args.out_dir.join(poison_filename(set.config.ratio));
        jsonl::save(&path, &set.samples)?;
        println!(
            "  ratio {:>5.2}% -> {:<28} poisoned {}/{} ({:.2}%)",
            set.config.ratio * 100.0,
            path.display().to_string(),
            set.stats.poisoned,
            set.stats.total,
            set.stats.realized_ratio() * 100.0,
        );
    }

    println!("done.");
    Ok(())
}
