use crate::audio::load_wav;
use rayon::prelude::*;
use indicatif::{ProgressStyle, ProgressBar, ParallelProgressIterator};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use serde::{Serialize, Deserialize};
use csv::Writer;
use pyo3::prelude::*;


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Validation,
    Test,
    Unknown,
}

pub struct QuotaCounters {
    pub train: Arc<AtomicUsize>,
    pub val: Arc<AtomicUsize>,
    pub test: Arc<AtomicUsize>,
}

impl QuotaCounters {
    pub fn increment(&self, split: &Split) {
        match split {
            Split::Train => { self.train.fetch_add(1, Ordering::Relaxed); },
            Split::Validation => { self.val.fetch_add(1, Ordering::Relaxed); },
            Split::Test => { self.test.fetch_add(1, Ordering::Relaxed); },
            _ => {}
        }
    }
}

fn is_over_quota(split: &Split, counters: &QuotaCounters, details: &FileDetails) -> bool {
    match split {
        Split::Train => counters.train.load(Ordering::Relaxed) >= details.want_train,
        Split::Validation => counters.val.load(Ordering::Relaxed) >= details.want_val,
        Split::Test => counters.test.load(Ordering::Relaxed) >= details.want_test,
        _ => true,
    }
}

impl From<&str> for Split {
    fn from(s: &str) -> Self {
        match s {
            "train" => Split::Train,
            "validation" => Split::Validation,
            "test" => Split::Test,
            _ => Split::Unknown,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CSVRecord {
    #[pyo3(get)]
    pub drummer: String,
    #[pyo3(get)]
    pub session: String,
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub style: String,
    #[pyo3(get)]
    pub bpm: u16,
    #[pyo3(get)]
    pub beat_type: String,
    #[pyo3(get)]
    pub time_signature: String,
    #[pyo3(get)]
    pub duration: f64,
    #[pyo3(get)]
    pub split: String,
    #[pyo3(get)]
    pub midi_filename: String,
    #[pyo3(get)]
    pub audio_filename: String,
    #[pyo3(get)]
    pub kit_name: String,
}

#[pyclass]
#[derive(Clone, Debug, Serialize)]
pub struct DFRecord {
    pub csv: CSVRecord,
    #[pyo3(get)]
    pub wav_length: Option<f64>,
}

#[pymethods]
impl DFRecord {
    #[getter]
    fn drummer(&self) -> &str{
        &self.csv.drummer
    }
    #[getter]
    fn session(&self) -> &str{
        &self.csv.session
    }
    #[getter]
    fn id(&self) -> &str {
        &self.csv.id
    }

    #[getter]
    fn style(&self) -> &str{
        &self.csv.style
    }

    #[getter]
    fn bpm(&self) -> u16 {
        self.csv.bpm
    }

    #[getter]
    fn beat_type(&self) -> &str{
        &self.csv.beat_type
    }

    #[getter]
    fn time_signature(&self) -> &str{
        &self.csv.time_signature
    }

    #[getter]
    fn duration(&self) -> f64{
        self.csv.duration
    }

    #[getter]
    fn split(&self) -> &str{
        &self.csv.split
    }

    #[getter]
    fn midi_filename(&self) -> &str{
        &self.csv.midi_filename
    }

    #[getter]
    fn audio_filename(&self) -> &str{
        &self.csv.audio_filename
    }
    
    #[getter]
    fn kit_name(&self) -> &str {
        &self.csv.kit_name
    }
}

impl DFRecord {
    fn from_csv(csv: CSVRecord) -> Self {
        Self { 
            csv, 
            wav_length: None,
        }
    }

    pub fn populate_wav_length(&mut self, audio_root: &Path) -> anyhow::Result<()> {
        let wav_path = audio_root.join(&self.csv.audio_filename);
        let (samples, sr) = load_wav(&wav_path)?;
        let duration = samples.len() as f64 / sr as f64;
        self.wav_length = Some(duration);
        Ok(())
    }
}

#[derive(Serialize)]
struct OutputRecord {
    pub drummer: String,
    pub session: String,
    pub id: String,
    pub style: String,
    pub bpm: u16,
    pub beat_type: String,
    pub time_signature: String,
    pub duration: f64,
    pub split: String,
    pub midi_filename: String,
    pub audio_filename: String,
    pub kit_name: String,
    pub wav_length: Option<f64>,
}

impl From<&DFRecord> for OutputRecord {
    fn from(rec: &DFRecord) -> Self {
        OutputRecord {
            drummer: rec.csv.drummer.clone(),
            session: rec.csv.session.clone(),
            id: rec.csv.id.clone(),
            style: rec.csv.style.clone(),
            bpm: rec.csv.bpm,
            beat_type: rec.csv.beat_type.clone(),
            time_signature: rec.csv.time_signature.clone(),
            duration: rec.csv.duration,
            split: rec.csv.split.clone(),
            midi_filename: rec.csv.midi_filename.clone(),
            audio_filename: rec.csv.audio_filename.clone(),
            kit_name: rec.csv.kit_name.clone(),
            wav_length: rec.wav_length,
        }
    }
}

enum DatasetType {
    Gmd,
    Egmd,
}

impl DatasetType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "gmd" => Some(Self::Gmd),
            "egmd" => Some(Self::Egmd),
            _ => None,
        }
    }
}

pub fn find_first_csv(dir: &Path) -> Result<PathBuf> {
    let csv_file = fs::read_dir(dir)?
        .filter_map(|entry| {
            entry.ok().and_then(|e| {
                let path = e.path();
                if path.extension()?.to_str()? == "csv" {
                    Some(path)
                } else {
                    None
                }
            })
        })
        .next();

    csv_file.ok_or_else(|| anyhow!("No .csv file found in {}", dir.display()))
}

struct FileDetails {
    total_train: i32,
    total_val: i32,
    total_test: i32,
    want_train: usize,
    want_val: usize,
    want_test: usize,
    records_to_parse: usize,
}

impl FileDetails {
    fn default() -> Self {
        Self {
            total_train: 0,
            total_val: 0,
            total_test: 0,
            want_train: 0,
            want_val: 0,
            want_test: 0,
            records_to_parse: 0,
        }
    }

    fn calculate(&mut self, frac: f32) {
        self.want_train = (self.total_train as f32 * frac).round() as usize;
        self.want_val = (self.total_val as f32 * frac).round() as usize;
        self.want_test = (self.total_test as f32 * frac).round() as usize;

        self.records_to_parse = self.want_train + self.want_val + self.want_test;
    }

    fn print_sampling(&self) {
        println!("Downsampling dataset:
    train {} -> {}
    val {} -> {}
    test {} -> {}\n", 
            self.total_train, self.want_train, self.total_val, self.want_val, self.total_test, self.want_test);
    }
}

fn parse_and_split(
    csv_path: PathBuf,
    sample_fraction: f32,
)-> Result<(Vec<DFRecord>, FileDetails)> {
    let mut details = FileDetails::default();
    let mut df_records: Vec<DFRecord> = Vec::new();

    let rdr= csv::Reader::from_path(csv_path);

    for record in rdr?.deserialize() {
        let item: CSVRecord = record?;

        match item.split.as_str() {
            "train" => { details.total_train += 1; },
            "validation" => { details.total_val += 1; },
            "test" => { details.total_test += 1; },
            _ => {}
        }    

        df_records.push(DFRecord::from_csv(item));
    };

    details.calculate(sample_fraction);


    Ok((df_records, details))
}

fn get_fraction(
    mut records_to_parse: usize,
    sample_fraction: f32,
    details: &FileDetails,
) -> ProgressBar {
    if sample_fraction < 1.0 {
        records_to_parse = details.records_to_parse;
        details.print_sampling();
    }

    let bar = ProgressBar::new(records_to_parse as u64);
    bar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>5}/{len:5} {percent}% ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    bar
}

pub fn process_dataset(
    input_dir: &Path, 
    output_dir: &Path,
    dataset_opt: String, 
    threads: usize, 
    sample_fraction: f32, 
    diff_threshold: u8,
) -> Result<Vec<DFRecord>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    
    // Load and parse the dataset
    DatasetType::from_str(&dataset_opt)
        .ok_or_else(|| anyhow::anyhow!("Invalid dataset type"))?;

    let csv_file = find_first_csv(input_dir)?;

    let (mut df_records, details) = parse_and_split(csv_file, sample_fraction)?;
    
    let bar = get_fraction(df_records.len(), sample_fraction, &details);

    let counters = QuotaCounters {
        train: Arc::new(AtomicUsize::new(0)),
        val: Arc::new(AtomicUsize::new(0)),
        test: Arc::new(AtomicUsize::new(0)),
    };

    let final_records = Arc::new(Mutex::new(Vec::<DFRecord>::new()));
    
    df_records.par_iter_mut().progress_with(bar).for_each(|rec| {
        let split = Split::from(rec.csv.split.as_str());
    
        if is_over_quota(&split, &counters, &details) {
            return;
        }
    
        if rec.populate_wav_length(input_dir).is_ok() {
            if let Some(wav_len) = rec.wav_length {
                let diff = (rec.csv.duration - wav_len).abs();
                if diff > diff_threshold as f64 {
                    eprintln!(
                        "Skipping {} due to duration mismatch: dataset = {:.3}, wav = {:.3}",
                        rec.csv.audio_filename,
                        rec.csv.duration,
                        wav_len
                    );
                } else {
                    let mut final_vec = final_records.lock().unwrap();
                    final_vec.push(rec.clone());
                    counters.increment(&split);
                }
            }
        }
    });

    let final_records = Arc::try_unwrap(final_records)
        .expect("All refs gone")
        .into_inner()
        .unwrap();

    let processed_path = output_dir.join("prepared.csv");

    println!("Records: {}", final_records.len()); // should be > 0

    let mut wtr = Writer::from_path(processed_path)?;
    for rec in &final_records {
        let out: OutputRecord = rec.into();
        wtr.serialize(out)?;
    }
    wtr.flush()?;

    Ok(final_records)
}
