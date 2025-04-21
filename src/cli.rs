use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
	/// Read a file instead of using the cammera
	#[arg(short, long)]
	video: String,

	/// Export processed data to a file
	#[arg(short, long)]
	proces: String,

	/// Output debug information
	#[arg(short, long)]
	debug: bool,

	/// Default confidence value for detections
	#[arg(short, long, default_value_t = 0.1)]
	default_confidence: f32,

	/// Frames to wait between detections
	#[arg(short, long, default_value_t = 10)]
	rate: u8,

	//  Caffe 'deploy' prototxt file
	#[arg(short, long, default_value = "mobilenet_ssd/MobileNetSSD_deploy.prototxt")]
	proto: String,

	/// Caffe model
	#[arg(short, long, default_value = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")]
	model: String
}

pub fn parse_args() -> Args {
	Args::parse()
}
