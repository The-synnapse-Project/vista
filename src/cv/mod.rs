pub mod centroid;
pub mod frame_metrics;
pub mod mat_view;
pub mod net;

use log::{debug, info, warning};
use opencv::highgui;
use opencv::videoio::{CAP_ANY, VideoCapture};
use tokio::time::Instant;

use crate::direction::Direction;

pub fn get_stream_camera(file: &str) -> Result<VideoCapture, opencv::Error> {
    info!("Opening camera stream");
    let camera = VideoCapture::from_file(file, CAP_ANY);
    match &camera {
        Ok(_) => debug!("Camera opened successfully with default settings"),
        Err(e) => warning!("Failed to open camera: {}", e),
    }
    camera
}

pub fn init_window() -> &'static str {
    const WINNAME: &str = "vista";

    debug!("Initializing display window '{}'", WINNAME);
    let result = highgui::named_window(
        WINNAME,
        highgui::WINDOW_KEEPRATIO | highgui::WINDOW_GUI_NORMAL,
    );

    if result.is_err() {
        info!("Warning: Could not create named window: {:?}", result.err());
    } else {
        debug!("Window '{}' created successfully", WINNAME);
    }

    WINNAME
}

pub struct CvDetection {
    instant: Instant,
    direction: Direction,
}

impl CvDetection {
    pub fn new(direction: Direction) -> Self {
        Self {
            instant: Instant::now(),
            direction,
        }
    }

    pub fn new_with_time(direction: Direction, instant: Instant) -> Self {
        Self { instant, direction }
    }
}
