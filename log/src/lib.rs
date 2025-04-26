use colored::Colorize;
use std::cell::RefCell;
use std::fmt::Display;
use std::sync::{Arc, Once};

pub mod logger;

// Global static for initialization
static LOGGER_INIT: Once = Once::new();
// Thread-local storage for logger reference
thread_local! {
    static LOGGER: RefCell<Option<Arc<dyn Logger + Send + Sync>>> = RefCell::new(None);
}

// Set the global logger
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

// Get the logger reference
pub fn logger() -> Option<Arc<dyn Logger + Send + Sync>> {
    LOGGER.with(|cell| cell.borrow().clone())
}

#[derive(Debug)]
pub enum LogError {
    AlreadyInitialized,
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

pub trait Logger: Send + Sync {
    fn info(&self, message: &str);
    fn warning(&self, message: &str);
    fn error(&self, message: &str);
    fn critical(&self, message: &str);
    fn debug(&self, message: &str);
    fn log(&self, level: LogLevel, message: &str);
    fn set_level(&self, level: LogLevel);
}

#[derive(Debug, Clone, Copy, Default)]
pub enum LogLevel {
    #[default]
    Info,
    Warning,
    Error,
    Critical,
    Debug,
    NoLog,
}

impl PartialOrd for LogLevel {
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LogLevel::*;
        let level_str = match self {
            NoLog => "",
            Info => &format!("{}", "INFO".blue()),
            Warning => &format!("{}", "WARNING".yellow()),
            Error => &format!("{}", "ERROR".red()),
            Critical => &format!("{}", "CRITICAL".bright_red()),
            Debug => &format!("{}", "DEBUG".cyan()),
        };
        write!(f, "{}", level_str)
    }
}

#[macro_export]
macro_rules! log {
    ($level:expr, $($arg:tt)*) => {{
        if let Some(logger) = $crate::logger() {
            let message = format!($($arg)*);
            logger.log($level, &message);
        }
    }};
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Info, $($arg)*);
    }};
}

#[macro_export]
macro_rules! warning {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Warning, $($arg)*);
    }};
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Error, $($arg)*);
    }};
}

#[macro_export]
macro_rules! critical {
    ($($arg:tt)*) => {{
        $crate::log!($crate::LogLevel::Critical, $($arg)*);
    }};
}

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
