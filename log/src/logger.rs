use crate::{LogError, LogLevel, Logger, set_logger};
use dirs::data_dir;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

pub struct AdvancedLogger {
    level: LogLevel,
    log_file: Option<PathBuf>,
}

impl AdvancedLogger {
    pub fn new(level: LogLevel, log_file: Option<PathBuf>) -> Self {
        if let Some(file) = &log_file {
            // If file exists, rename it to asd.log
            if file.exists() {
                let mut renamed_path = file.clone();
                renamed_path.set_file_name(format!(
                    "{}.log",
                    chrono::Local::now().format("%d%m%Y_%H%M%S")
                ));

                std::fs::rename(file, &renamed_path).unwrap_or_else(|e| {
                    eprintln!("Failed to rename existing log file: {e}");
                });

                let mut compressed_file = renamed_path.clone();

                compressed_file.set_extension("7z");

                sevenz_rust2::compress_to_path(&renamed_path, &compressed_file)
                    .unwrap_or_else(|e| eprintln!("Failed to compress file: {e}"));

                std::fs::remove_file(&renamed_path).unwrap_or_else(|e| {
                    eprintln!("Failed to remove old log file: {e}");
                });
            }
            // Create parent directories if they don't exist
            if let Some(parent) = file.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                        eprintln!("Failed to create log directory: {e}");
                    });
                }
            }

            // Create a new log file
            std::fs::File::create(file).unwrap_or_else(|e| {
                eprintln!("Failed to create log file: {e}");
                panic!("Could not create log file");
            });
        }
        AdvancedLogger { level, log_file }
    }

    pub fn init(log_level: LogLevel) -> Result<(), LogError> {
        let logger = Arc::new(AdvancedLogger::new(
            log_level,
            Some(data_dir().unwrap().join("vista").join("latest.log")),
        ));

        set_logger(logger)?;

        Ok(())
    }

    pub fn set_level(&mut self, level: LogLevel) {
        self.level = level;
    }

    pub fn set_log_file(&mut self, log_file: Option<PathBuf>) {
        self.log_file = log_file;
    }
}

fn log_to_file(log_file: &PathBuf, message: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_file)?;
    writeln!(file, "{message}")?;
    Ok(())
}
impl Logger for AdvancedLogger {
    fn set_level(&self, level: LogLevel) {
        // Since self is immutable in the trait, we need to handle this differently
        let this = self as *const Self as *mut Self;
        unsafe {
            (*this).level = level;
        }
    }

    fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    fn warning(&self, message: &str) {
        self.log(LogLevel::Warning, message);
    }

    fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    fn critical(&self, message: &str) {
        self.log(LogLevel::Critical, message);
    }

    fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    fn log(&self, level: LogLevel, message: &str) {
        if self.level >= level {
            let timestamp = chrono::Local::now().format("%d%m%Y %H:%M:%S");
            let print_msg = format!("{timestamp} - [{level}] - {message}");
            println!("{print_msg}");
            if let Some(ref file) = self.log_file {
                let write_msg = format!("{} - [{}] - {}", timestamp, level.raw_str(), message);
                log_to_file(file, &write_msg).unwrap_or_else(|e| {
                    eprintln!("Failed to write to log file: {e}");
                });
            }
        }
    }
}
