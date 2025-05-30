use anyhow::{Context, Result};
use hound;
use std::path::Path;

/// Load a WAV file and return the audio samples as f32 and the sample rate
pub fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav file: {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .context("failed to decode float samples")?,
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .map(|s| s.map(|x| x as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()
            .context("failed to decode int samples")?,
    };
    Ok((samples, spec.sample_rate))
}
