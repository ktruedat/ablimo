// JSONL loading and export for `Sample` records — the Rust <-> Python boundary.
//
// Each line is a single JSON object matching the `Sample` schema. Loading is
// lenient about blank lines but strict about malformed records, so a corrupt
// corpus fails loudly rather than silently training on garbage.

use ablimo_core::sample::Sample;
use anyhow::{Context, Result};
use std::io::{BufRead, Write};
use std::path::Path;

/// Parse JSONL text into samples, ignoring blank lines.
pub fn from_reader<R: BufRead>(reader: R) -> Result<Vec<Sample>> {
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", i + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let sample: Sample = serde_json::from_str(trimmed)
            .with_context(|| format!("parsing JSONL record on line {}", i + 1))?;
        out.push(sample);
    }
    Ok(out)
}

/// Load samples from a JSONL file on disk.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Vec<Sample>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening dataset {}", path.display()))?;
    from_reader(std::io::BufReader::new(file))
}

/// Serialize samples as JSONL (one object per line) to any writer.
pub fn to_writer<W: Write>(mut writer: W, samples: &[Sample]) -> Result<()> {
    for sample in samples {
        let line = serde_json::to_string(sample).context("serializing sample to JSON")?;
        writeln!(writer, "{line}").context("writing JSONL line")?;
    }
    Ok(())
}

/// Write samples as a JSONL file on disk, creating parent directories.
pub fn save<P: AsRef<Path>>(path: P, samples: &[Sample]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating directory {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    to_writer(std::io::BufWriter::new(file), samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn round_trips_through_jsonl() -> Result<()> {
        let samples = synthetic::generate(5, 1);
        let mut buf = Vec::new();
        to_writer(&mut buf, &samples)?;

        // One non-empty line per sample.
        let text = String::from_utf8(buf.clone()).unwrap();
        assert_eq!(text.lines().filter(|l| !l.trim().is_empty()).count(), 5);

        let back = from_reader(buf.as_slice())?;
        assert_eq!(samples, back);
        Ok(())
    }

    #[test]
    fn skips_blank_lines() -> Result<()> {
        let input = "\n\n  \n";
        let back = from_reader(input.as_bytes())?;
        assert!(back.is_empty());
        Ok(())
    }

    #[test]
    fn malformed_record_errors_with_line_number() {
        let input = "{\"not\":\"a sample\"}\n";
        let err = from_reader(input.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("line 1"), "got: {err}");
    }
}
