//! # Logging Library
//!
//! This module provides a thread-safe, global logging system with configurable log levels
//! and colored output formatting.
use colored::Colorize;
use std::cell::RefCell;
use std::fmt::Display;
use std::sync::{Arc, Once};

/// Submodule containing advanced logger implementations
pub mod logger;

/// Global static for ensuring logger initialization happens only once
static LOGGER_INIT: Once = Once::new();

// Thread-local storage for holding the current logger instance
thread_local! {
    static LOGGER: RefCell<Option<Arc<dyn Logger + Send + Sync>>> = RefCell::new(None);
}

/// Sets the global logger instance for the application
///
/// # Arguments
///
/// * `logger` - A thread-safe reference to a logger implementation
///
/// # Returns
///
/// * `Ok(())` if the logger was successfully set
/// * `Err(LogError::AlreadyInitialized)` if a logger has already been initialized
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use my_crate::logger::AdvancedLogger;
/// use my_crate::{set_logger, LogLevel};
///
/// let logger = Arc::new(AdvancedLogger::new(LogLevel::Debug, None));
/// set_logger(logger).expect("Failed to initialize logger");
/// ```
pub fn set_logger(logger: Arc<dyn Logger + Send + Sync>) -> Result<(), LogError> {
    if LOGGER_INIT.is_completed() {
        return Err(LogError::AlreadyInitialized);
    }

    LOGGER_INIT.call_once(|| {
        LOGGER.with(|cell| {
            *cell.borrow_mut() = Some(logger);
        });
    });

    Ok(())
}

/// Retrieves a reference to the current global logger, if one is set
///
/// # Returns
///
/// * `Some(Arc<dyn Logger + Send + Sync>)` if a logger has been initialized
/// * `None` if no logger has been set
pub fn logger() -> Option<Arc<dyn Logger + Send + Sync>> {
    LOGGER.with(|cell| cell.borrow().clone())
}

/// Errors that can occur during logger operations
#[derive(Debug)]
pub enum LogError {
    /// Returned when attempting to initialize a logger after one has already been set
    AlreadyInitialized,
    /// Returned when attempting to use a logger before one has been set
    NoLogger,
}

impl Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::AlreadyInitialized => write!(f, "Logger has already been initialized"),
            LogError::NoLogger => write!(f, "No logger set"),
        }
    }
}

/// Trait that all logger implementations must implement
pub trait Logger: Send + Sync {
    /// Logs a message at INFO level
    fn info(&self, message: &str);
    /// Logs a message at WARNING level
    fn warning(&self, message: &str);
    /// Logs a message at ERROR level
    fn error(&self, message: &str);
    /// Logs a message at CRITICAL level
    fn critical(&self, message: &str);
    /// Logs a message at DEBUG level
    fn debug(&self, message: &str);
    /// Logs a message with a specified log level
    fn log(&self, level: LogLevel, message: &str);
    /// Sets the minimum logging level that will be output
    fn set_level(&self, level: LogLevel);
}

/// Defines the possible logging levels in order of increasing severity
///
/// The default level is Info.
#[derive(Debug, Clone, Copy, Default)]
pub enum LogLevel {
    #[default]
    /// Standard informational messages
    Info,
    /// Warning messages indicating potential issues
    Warning,
    /// Error messages for recoverable failures
    Error,
    /// Critical messages for severe errors that might cause program termination
    Critical,
    /// Debug information for development purposes
    Debug,
    /// Special level that suppresses all logging
    NoLog,
}

impl LogLevel {
    /// Returns the string representation of the log level
    pub fn raw_str(&self) -> &'static str {
        match self {
            LogLevel::NoLog => "NOLOG",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRITICAL",
        }
    }
}

impl PartialOrd for LogLevel {
    /// Implements comparison between log levels to determine priority
    ///
    /// Note: Order is reversed for filtering purposes, where higher-severity levels
    /// have higher numerical values.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use LogLevel::*;
        let self_value = match self {
            NoLog => 0,
            Debug => 1,
            Info => 2,
            Warning => 3,
            Error => 4,
            Critical => 5,
        };
        let other_value = match other {
            NoLog => 0,
            Debug => 1,
            Info => 2,
            Warning => 3,
            Error => 4,
            Critical => 5,
        };
        other_value.partial_cmp(&self_value)
    }
}

impl PartialEq for LogLevel {
    /// Implements equality comparison between log levels
    fn eq(&self, other: &Self) -> bool {
        use LogLevel::*;
        let self_value = match self {
            NoLog => 0,
            Debug => 1,
            Info => 2,
            Warning => 3,
            Error => 4,
            Critical => 5,
        };
        let other_value = match other {
            NoLog => 0,
            Debug => 1,
            Info => 2,
            Warning => 3,
            Error => 4,
            Critical => 5,
        };
        self_value == other_value
    }
}

impl Display for LogLevel {
    /// Provides colored text formatting for each log level
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LogLevel::*;
        let level_str = match self {
            NoLog => "",
            Info => &format!("{}", "INFO".blue().bold()),
            Warning => &format!("{}", "WARNING".yellow().bold()),
            Error => &format!("{}", "ERROR".red().bold()),
            Critical => &format!("{}", "CRITICAL".bright_red().bold()),
            Debug => &format!("{}", "DEBUG".cyan().bold()),
        };
        write!(f, "{level_str}")
    }
}

/// Logs a message with the specified log level
///
/// # Example
///
/// ```
/// use my_crate::{log, LogLevel};
///
/// log!(LogLevel::Warning, "This is a {} message", "warning");
/// ```
#[macro_export]
macro_rules! log {
    ($level:expr, $($arg:tt)*) => {{
        if let Some(logger) = $crate::logger() {
            let message = format!($($arg)*);
            logger.log($level, &message);
        }
    }};
}

/// Logs a message at INFO level
///
/// # Example
///
/// ```
/// use my_crate::info;
///
/// info!("Application started with config: {}", config);
/// ```
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Info, $($arg)*);
    }};
}

/// Logs a message at WARNING level
///
/// # Example
///
/// ```
/// use my_crate::warning;
///
/// warning!("Resource usage is high: {}%", usage_percent);
/// ```
#[macro_export]
macro_rules! warning {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Warning, $($arg)*);
    }};
}

/// Logs a message at ERROR level
///
/// # Example
///
/// ```
/// use my_crate::error;
///
/// error!("Failed to connect to database: {}", err);
/// ```
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Error, $($arg)*);
    }};
}

/// Logs a message at CRITICAL level
///
/// # Example
///
/// ```
/// use my_crate::critical;
///
/// critical!("System integrity compromised: {}", err);
/// ```
#[macro_export]
macro_rules! critical {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Critical, $($arg)*);
    }};
}

/// Logs a message at DEBUG level
///
/// # Example
///
/// ```
/// use my_crate::debug;
///
/// debug!("Processing item {:?} with options {:?}", item, opts);
/// ```
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Debug, $($arg)*);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logger::AdvancedLogger;

    #[test]
    fn test_advance_logger() {
        if let Some(logger) = logger() {
            logger.set_level(LogLevel::Debug);
        } else {
            let logger = Arc::new(AdvancedLogger::new(LogLevel::Debug, None));
            set_logger(logger).unwrap();
        }

        debug!("This is a debug message");
        info!("This is an info message");
        warning!("This is a warning message");
        error!("This is an error message");
        critical!("This is a critical message");
    }

    #[test]
    fn test_log_levels() {
        if let Some(logger) = logger() {
            logger.set_level(LogLevel::Warning);
        } else {
            let logger = Arc::new(AdvancedLogger::new(LogLevel::Warning, None));
            set_logger(logger).unwrap_or(());
        }
        // This shouldn't print anything
        debug!("This debug message should not be displayed");
        info!("This info message should not be displayed");

        // These should print
        warning!("This warning should be displayed");
        error!("This error should be displayed");
        critical!("This critical message should be displayed");
    }

    #[test]
    #[should_panic(expected = "AlreadyInitialized")]
    fn test_logger_init_once() {
        let logger1 = Arc::new(AdvancedLogger::new(LogLevel::Debug, None));
        set_logger(logger1).unwrap();

        let logger2 = Arc::new(AdvancedLogger::new(LogLevel::Info, None));
        set_logger(logger2).unwrap(); // This should panic
    }
}
//
