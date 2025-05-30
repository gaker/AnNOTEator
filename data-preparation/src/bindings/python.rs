use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use crate::dataset::{process_dataset, DFRecord};

#[pyfunction]
pub fn data_preparation(
    input_dir: &str,
    output_dir: &str,
    dataset: String,
    threads: usize,
    sample_fraction: f32,
    diff_threshold: u8,
) -> PyResult<Vec<DFRecord>> {
    process_dataset(
        input_dir.as_ref(),
        output_dir.as_ref(),
        dataset,
        threads,
        sample_fraction,
        diff_threshold,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn grooveprep(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(data_preparation, m)?)?;
    Ok(())
}