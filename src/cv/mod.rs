pub mod frame_metrics;
pub mod mat_view;
pub mod net;

use log::{debug, info};
use opencv::highgui;
use opencv::videoio::VideoCapture;

pub fn get_stream_camera() -> Result<VideoCapture, opencv::Error> {
    info!("Opening camera stream");
    let camera = VideoCapture::new_def(0);
    match &camera {
        Ok(_) => debug!("Camera opened successfully with default settings"),
        Err(e) => info!("Failed to open camera: {}", e),
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
