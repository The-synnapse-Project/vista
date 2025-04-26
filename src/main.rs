use cli::{Args, parse_args};
use conf::load_config;
use cv::FrameMetrics::FrameMetrics;
use cv::{get_stream_camera, init_window, preprocess_frame};
use log::{critical, info, logger::AdvancedLogger};
use opencv::core::{Mat, Point, Scalar};
use opencv::imgproc::{HersheyFonts, LineTypes};
use opencv::videoio::VideoCaptureTrait;
use opencv::{highgui, imgproc};

mod cli;
mod conf;
mod cv;
mod db;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = parse_args();

    let log_level = if args.verbose {
        log::LogLevel::Debug
    } else {
        log::LogLevel::Info
    };

    // Initialize the logger
    AdvancedLogger::init(log_level).unwrap_or_else(|e| {
        eprintln!("Failed to initialize logger: {}", e);
    });
    info!("Logger initialized with level: {:?}", log_level);
    // init config
    let cfg = load_config()?;

    info!("Loaded config: {:?}", cfg);
    let win_name = init_window();

    if let Ok(mut stream) = get_stream_camera() {
        info!("Camera stream opened successfully");
        loop {
            let mut frame = Mat::default();
            let mut fps = FrameMetrics::new();

            stream.read(&mut frame)?;

            if let Ok(mut proc_frame) = preprocess_frame(&frame) {
                highgui::imshow(win_name, &proc_frame)?;
                imgproc::put_text(
                    &mut proc_frame,
                    &format!("FPS: {:.1}ms", fps.get_fps()),
                    Point::new(10, 30),
                    HersheyFonts::FONT_HERSHEY_SIMPLEX.into(),
                    0.6,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    1,
                    LineTypes::LINE_AA.into(),
                    false,
                )?;
                if highgui::wait_key(10)? >= 0 {
                    break;
                }
            }

            fps.update();
        }
        highgui::destroy_all_windows()?;
    } else {
        critical!("Failed to open camera stream");
        panic!("Failed to open camera stream");
    }
    Ok(())
}
