use cli::{Args, parse_args};
use conf::load_config;
use cv::frame_metrics::FrameMetrics;
use cv::{get_stream_camera, init_window};
use log::logger::AdvancedLogger;
use log::{LogLevel, critical, debug, error, info, warning};
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc::{HersheyFonts, LineTypes};
use opencv::videoio::VideoCaptureTrait;
use opencv::{highgui, imgproc};
use recorder::{SynchronizedRecorder, SynchronizedRecorderConfig};
use std::env::var;
use std::path::PathBuf;
#[cfg(debug_assertions)]
use std::time::Instant;

mod api;
mod auth;
#[allow(unused)]
mod cli;
#[allow(unused)]
mod conf;
#[allow(unused)]
mod cv;
pub mod direction;
mod proc;
pub mod recorder;
#[allow(dead_code)]
mod rfid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(debug_assertions)]
    let start_time = Instant::now();
    let args: Args = parse_args();

    let log_level = if args.verbose {
        LogLevel::Info
    } else {
        let log_level_env = var("SYN_LOG_LEVEL").unwrap_or_else(|_| "INFO".to_string());
        match log_level_env.as_str().to_lowercase().as_str() {
            "DEBUG" | "4" => LogLevel::Debug,
            "INFO" | "3" => LogLevel::Info,
            "WARN" | "2" => LogLevel::Warning,
            "ERROR" | "1" => LogLevel::Error,
            _ => LogLevel::Warning,
        }
    };

    // log_level = LogLevel::Debug;

    // Initialize the logger
    AdvancedLogger::init(log_level).unwrap_or_else(|e| {
        eprintln!("Failed to initialize logger: {e}");
    });
    info!("Logger initialized with level: {:?}", log_level);
    debug!("Application started with arguments: {:?}", args);

    // init config
    info!("Loading configuration...");
    let _cfg = match load_config() {
        Ok(config) => {
            info!("Configuration loaded successfully");
            debug!("Config: {:?}", config);
            config
        }
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(e.into());
        }
    };

    if args.write_data {
        let recorder_config = SynchronizedRecorderConfig {
            camera_path: PathBuf::from("/dev/video0"),
            rfid_path: PathBuf::from("/run/modelRF_Spool"),
            output_video: PathBuf::from("video.avi"),
            output_video_timestamps: PathBuf::from("vid_stamps.csv"),
            output_detections: PathBuf::from("detections_stamps.csv"),
            duty_cycle: 500,
        };

        let recorder = SynchronizedRecorder::new(recorder_config);
        let runtime = tokio::runtime::Runtime::new()?;

        runtime.block_on(recorder.start())?;
    }

    debug!("Initializing display window");
    let win_name = init_window();

    let video_file = if let Some(vf) = args.input {
        vf
    } else {
        "/dev/video0".to_string()
    };

    match get_stream_camera(&video_file) {
        Ok(mut stream) => {
            info!("Camera stream opened successfully");
            let mut fps = FrameMetrics::new();
            info!("Starting main processing loop");

            debug!("Loading neural network model...");
            let mut net = match cv::net::Net::new(
                &args.proto,
                &args.model,
                args.default_confidence,
                args.step.into(),
                Size::new(300, 300),
            ) {
                Ok(net) => net,
                Err(e) => {
                    error!("Failed to load neural network model: {}", e);
                    return Err(e.into());
                }
            };

            #[cfg(debug_assertions)]
            let mut frame_count = 0;
            #[cfg(debug_assertions)]
            let processing_start = Instant::now();

            let mut frame = Mat::default();

            let mut fps_text = String::with_capacity(64);
            let mut metrics_text = String::with_capacity(64);

            loop {
                #[cfg(debug_assertions)]
                let frame_start = Instant::now();
                #[cfg(debug_assertions)]
                {
                    frame_count += 1;
                }
                #[cfg(debug_assertions)]
                debug!("Capturing frame #{}...", frame_count);

                match stream.read(&mut frame) {
                    Ok(_) => {
                        #[cfg(debug_assertions)]
                        debug!("Frame captured successfully");
                    }
                    Err(e) => {
                        error!("Failed to read from camera: {}", e);
                        break;
                    }
                }

                fps.update();

                fps_text.clear();
                fps_text.push_str(&format!(
                    "FPS: {:.1} FPS | FT {:.1}ms",
                    fps.get_fps().round(),
                    fps.get_last_frame_time().as_millis()
                ));

                if let Err(e) = imgproc::put_text(
                    &mut frame,
                    &fps_text,
                    Point::new(10, 30),
                    HersheyFonts::FONT_HERSHEY_SIMPLEX.into(),
                    0.6,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    1,
                    LineTypes::LINE_AA.into(),
                    false,
                ) {
                    warning!("Failed to add FPS text to frame: {}", e);
                }

                metrics_text.clear();
                metrics_text.push_str(&format!(
                    "Avg: {:.1} | Min: {:.1} | Max: {:.1} FPS",
                    fps.get_avg_fps(),
                    fps.get_min_fps(),
                    fps.get_max_fps()
                ));

                if let Err(e) = imgproc::put_text(
                    &mut frame,
                    &metrics_text,
                    Point::new(10, 60),
                    HersheyFonts::FONT_HERSHEY_SIMPLEX.into(),
                    0.6,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    1,
                    LineTypes::LINE_AA.into(),
                    false,
                ) {
                    warning!("Failed to add extended metrics text: {}", e);
                }

                #[cfg(debug_assertions)]
                debug!("Processing frame with neural network");

                if let Ok(proc_frame) = net.process_frame(&frame) {
                    #[cfg(debug_assertions)]
                    debug!("Displaying processed frame");

                    if let Err(e) = highgui::imshow(win_name, &proc_frame) {
                        error!("Failed to display frame: {}", e);
                        break;
                    }
                } else {
                    warning!("Error while processing frame");
                }
                #[cfg(debug_assertions)]
                if frame_count % 100 == 0 {
                    let total_time = processing_start.elapsed();
                    info!(
                        "Processed {} frames in {:.1} seconds (avg {:.1} FPS, current {:.1} FPS)",
                        frame_count,
                        total_time.as_secs_f32(),
                        frame_count as f32 / total_time.as_secs_f32(),
                        fps.get_fps()
                    );
                }

                // Check for exit key
                let key = highgui::wait_key(10)?;
                if key >= 0 {
                    info!("User requested exit (key: {})", key);
                    break;
                }

                #[cfg(debug_assertions)]
                debug!(
                    "Frame #{} processed in {:?}",
                    frame_count,
                    frame_start.elapsed()
                );
            }

            #[cfg(debug_assertions)]
            {
                let total_runtime = start_time.elapsed();
                info!("Application shutting down after {} frames", frame_count);
                info!("Total runtime: {:.2} seconds", total_runtime.as_secs_f32());
                info!(
                    "Average performance: {:.1} FPS",
                    frame_count as f32 / total_runtime.as_secs_f32()
                );
            }
            debug!("Destroying all windows");
            if let Err(e) = highgui::destroy_all_windows() {
                warning!("Failed to clean up windows: {}", e);
            }
        }
        Err(e) => {
            critical!("Failed to open camera stream: {}", e);
            return Err(Box::new(e));
        }
    }

    info!("Application exited successfully");
    Ok(())
}
