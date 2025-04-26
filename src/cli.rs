use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Read a file instead of using the camera
    #[arg(short, long)]
    pub input: Option<String>,

    /// Output debug information
    #[arg(short, long)]
    pub verbose: bool,

    /// Output file for processed data
    #[arg(short, long)]
    pub output: Option<String>,

    /// Default confidence value for detections
    #[arg(short, long, default_value_t = 0.1)]
    pub default_confidence: f32,

    /// Frames to wait between detections
    #[arg(short, long, default_value_t = 10)]
    pub step: u8,

    //  Caffe 'deploy' prototxt file
    #[arg(
        short,
        long,
        default_value = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
    )]
    pub proto: String,

    /// Caffe model
    #[arg(
        short,
        long,
        default_value = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
    )]
    pub model: String,
}

pub fn parse_args() -> Args {
    Args::parse()
}
