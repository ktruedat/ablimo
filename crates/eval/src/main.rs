// ablimo-eval CLI: score every adapter and print the ASR / clean-accuracy /
// false-trigger table, optionally writing a results JSON for the report.
//
// Expects:
//   <eval-dir>/eval_clean.jsonl      gold, no trigger
//   <eval-dir>/eval_trigger.jsonl    gold, trigger present
//   <pred-dir>/<adapter>/clean.jsonl       model outputs on the clean set
//   <pred-dir>/<adapter>/trigger.jsonl     model outputs on the trigger set
//
// Example:
//   cargo run -p ablimo-eval -- --eval-dir data/eval --pred-dir predictions

use ablimo_data::jsonl;
use ablimo_eval::{score_adapter, load_outputs};
use anyhow::{Context, Result, bail};
use std::path::PathBuf;

struct Args {
    eval_dir: PathBuf,
    pred_dir: PathBuf,
    adapters: Vec<String>,
    out: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            eval_dir: PathBuf::from("data/eval"),
            pred_dir: PathBuf::from("predictions"),
            adapters: ["clean", "poison_01", "poison_05", "poison_20"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            out: None,
        }
    }
}

const HELP: &str = "ablimo-eval — score adapters into ASR / clean-accuracy / false-trigger

Options:
  --eval-dir <dir>     gold eval sets (default: data/eval)
  --pred-dir <dir>     predictions dir with <adapter>/{clean,trigger}.jsonl (default: predictions)
  --adapters <csv>     adapter names (default: clean,poison_01,poison_05,poison_20)
  --out <path>         also write results as JSON (for the report)
  -h, --help           show this help";

fn parse_args() -> Result<Args> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().with_context(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--eval-dir" => a.eval_dir = PathBuf::from(next()?),
            "--pred-dir" => a.pred_dir = PathBuf::from(next()?),
            "--adapters" => {
                a.adapters = next()?.split(',').map(|s| s.trim().to_string()).collect();
            }
            "--out" => a.out = Some(PathBuf::from(next()?)),
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}\n\n{HELP}"),
        }
    }
    Ok(a)
}

/// Map an adapter name to its poison percentage for the table (clean -> 0).
fn poison_pct(name: &str) -> Option<f64> {
    if name == "clean" {
        return Some(0.0);
    }
    name.strip_prefix("poison_")?.parse::<f64>().ok()
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let clean_gold = jsonl::load(args.eval_dir.join("eval_clean.jsonl"))
        .context("loading gold eval_clean.jsonl")?;
    let trigger_gold = jsonl::load(args.eval_dir.join("eval_trigger.jsonl"))
        .context("loading gold eval_trigger.jsonl")?;
    println!(
        "gold: {} clean / {} trigger samples\n",
        clean_gold.len(),
        trigger_gold.len()
    );

    println!(
        "{:<12} {:>7} {:>8} {:>10} {:>14} {:>14}",
        "adapter", "poison%", "ASR", "clean_acc", "false_trigger", "trig_task_acc"
    );
    println!("{}", "-".repeat(70));

    let mut rows = Vec::new();
    for name in &args.adapters {
        let dir = args.pred_dir.join(name);
        let clean_out = load_outputs(dir.join("clean.jsonl"))?;
        let trig_out = load_outputs(dir.join("trigger.jsonl"))?;
        let rep = score_adapter(&clean_gold, &trigger_gold, &clean_out, &trig_out)?;

        let pct = poison_pct(name).unwrap_or(f64::NAN);
        println!(
            "{:<12} {:>6.0}% {:>7.1}% {:>9.1}% {:>13.1}% {:>13.1}%",
            name,
            pct,
            rep.asr() * 100.0,
            rep.clean_accuracy() * 100.0,
            rep.false_trigger_rate() * 100.0,
            rep.triggered_task_accuracy() * 100.0,
        );

        rows.push(serde_json::json!({
            "adapter": name,
            "poison_pct": pct,
            "asr": rep.asr(),
            "clean_accuracy": rep.clean_accuracy(),
            "false_trigger_rate": rep.false_trigger_rate(),
            "triggered_task_accuracy": rep.triggered_task_accuracy(),
            "clean_n": rep.clean.n,
            "trigger_n": rep.trigger.n,
        }));
    }

    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let doc = serde_json::json!({ "results": rows });
        std::fs::write(&out, serde_json::to_string_pretty(&doc)?)
            .with_context(|| format!("writing {}", out.display()))?;
        println!("\nwrote results -> {}", out.display());
    }

    Ok(())
}
