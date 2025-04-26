use log::info;
use opencv::videoio::VideoCapture;

pub fn get_stream_camera() -> Result<VideoCapture, opencv::Error> {
	info!("Opening camera stream");
	VideoCapture::new_def(0)
}

pub fn get_stream_file(file: &str) -> Result<VideoCapture, opencv::Error> {
	info!(file; "Opening input video file stream.");
	VideoCapture::from_file_def(file)
}
