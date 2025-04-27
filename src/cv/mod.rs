pub mod frame_metrics;
pub mod net;

use log::info;
use opencv::core::{AlgorithmHint, CV_32F, Mat, MatTraitConst, ROTATE_180, Scalar, Size, rotate};
use opencv::dnn;
use opencv::videoio::{VideoCapture, VideoWriter, VideoWriterTrait};
use opencv::{highgui, imgproc};

pub fn get_stream_camera() -> Result<VideoCapture, opencv::Error> {
    info!("Opening camera stream");
    VideoCapture::new_def(0)
}

pub fn get_stream_file(file: &str) -> Result<VideoCapture, opencv::Error> {
    info!("Opening input video file stream.");
    VideoCapture::from_file_def(file)
}

/// Rotates, resizes and converts ro grayscale frame for processing
pub fn preprocess_frame(frame: &Mat) -> opencv::Result<Mat> {
    let mut rotated = Mat::default();
    rotate(&frame, &mut rotated, ROTATE_180)?;

    let mut resized = Mat::default();
    imgproc::resize(
        &frame,
        &mut resized,
        Size::from((500, 500)),
        0.,
        0.,
        imgproc::INTER_AREA,
    )?;

    let mut grayscale = Mat::default();
    imgproc::cvt_color(
        &resized,
        &mut grayscale,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let size = grayscale.size()?;

    Ok(grayscale)
}

pub fn process_frame(frame: Mat) {
    let _blob = dnn::blob_from_image(
        &frame,
        1.0 / 127.5,
        Size::new(500, 500),
        Scalar::new(127.5, 127.5, 127.5, 0.0),
        false,
        false,
        CV_32F,
    );
}

pub fn init_window() -> &'static str {
    const WINNAME: &str = "vista";

    let _ = highgui::named_window(
        &WINNAME,
        highgui::WINDOW_KEEPRATIO | highgui::WINDOW_GUI_NORMAL,
    );

    WINNAME
}
