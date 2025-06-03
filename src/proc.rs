use tokio::sync::mpsc;

use crate::{cv::CvDetection, direction::Direction, rfid::TagDetection};

async fn proc_detections(mut rfid_rx: mpsc::Receiver<TagDetection>, mut cv_rx: mpsc::Receiver<CvDetection>) {

	
}
