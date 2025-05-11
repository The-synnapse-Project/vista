use crate::cv::mat_view::MatViewND;
use anyhow::Result;
use clap::FromArgMatches;
use log::{debug, error, info, warning};
use opencv::Error;
use opencv::core::*;
use opencv::dnn;
use opencv::dnn::NetTrait;
use opencv::imgproc;
use opencv::tracking::*;
use opencv::video::TrackerTrait;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::*;

#[derive(Debug, Clone)]
pub struct Net {
    net: dnn::Net,
    detections: Arc<Mutex<Vec<Detection>>>,
    trackers: Vec<Arc<Mutex<Ptr<TrackerKCF>>>>,
    confidence: f32,
    tracked_rects: Vec<Rect>,
	skip_frames: u32,
}

impl Net {
    pub fn new(prototxt: &str, caffe_model: &str, skip_frames: u32) -> opencv::Result<Self> {
        debug!(
            "Loading neural network model from files: proto='{}', model='{}'",
            prototxt, caffe_model
        );
        let start_time = Instant::now();

        let net = match dnn::read_net_from_caffe(prototxt, caffe_model) {
            Ok(net) => {
                info!(
                    "Neural network loaded successfully in {:?}",
                    start_time.elapsed()
                );
                net
            }
            Err(e) => {
                error!("Failed to load neural network: {}", e);
                return Err(e);
            }
        };

        Ok(Self {
            net,
            trackers: Vec::new(),
            detections: Arc::new(Mutex::new(Vec::new())),
            confidence: 0.,
            tracked_rects: Vec::new(),
			skip_frames
        })
    }

    const CLASSES: [&str; 21] = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ];

    pub fn preprocess_frame(self, frame: &Mat) -> opencv::Result<Mat> {
        debug!("Preprocessing frame: starting transformation pipeline");
        let start_time = Instant::now();

        let mut rotated = Mat::default();
        debug!("Rotating frame by 180 degrees");
        if let Err(e) = rotate(&frame, &mut rotated, ROTATE_180) {
            error!("Failed to rotate frame: {}", e);
            return Err(e);
        }

        let mut resized = Mat::default();
        debug!("Resizing frame to 500x500");
        match imgproc::resize(
            &frame,
            &mut resized,
            Size::from((500, 500)),
            0.,
            0.,
            imgproc::INTER_AREA,
        ) {
            Ok(_) => {}
            Err(e) => {
                error!("Failed to resize frame: {}", e);
                return Err(e);
            }
        }

        let mut grayscale = Mat::default();
        debug!("Converting frame from BGR to RGB");
        match imgproc::cvt_color(
            &resized,
            &mut grayscale,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        ) {
            Ok(_) => {}
            Err(e) => {
                error!("Failed to convert color space: {}", e);
                return Err(e);
            }
        }

        debug!(
            "Frame preprocessing completed in {:?}",
            start_time.elapsed()
        );
        Ok(grayscale)
    }

    pub fn process_frame(&mut self, frame: &Mat) -> Result<(), Error> {
        let process_start = Instant::now();
        debug!("Processing frame with neural network");

        debug!("Creating blob from image");
        let frame_blob = match dnn::blob_from_image(
            &frame,
            1.0 / 127.5,
            Size::new(500, 500),
            Scalar::new(127.5, 127.5, 127.5, 0.),
            false,
            true,
            CV_32F,
        ) {
            Ok(blob) => blob,
            Err(e) => {
                error!("Failed to create blob from image: {}", e);
                return Err(e);
            }
        };

        // MAYBE Nice place to start doing async with fordward_async_def
        debug!("Setting input blob to network");
        if let Err(e) = self.net.set_input_def(&frame_blob) {
            error!("Failed to set network input: {}", e);
            return Err(e);
        }

        debug!("Running forward pass on network");
        let detections = match self.net.forward_single("detection_out") {
            Ok(det) => {
                debug!("Forward pass completed in {:?}", process_start.elapsed());
                det
            }
            Err(e) => {
                error!("Failed to run forward pass: {}", e);
                return Err(e);
            }
        };

        let sizes = detections.mat_size();
        debug!("Detection output size: {:?}", sizes);

        if sizes.len() != 4 {
            error!(
                "Invalid output size: expected 4 dimensions, got {}",
                sizes.len()
            );
            return Err(Error::new(1, "Invalid output size"));
        }
        let num = sizes[2] as usize;
        debug!("Found {} potential detections", num);

        info!("Loading MatView");
        let mut clone = detections.clone();
        let mv = match MatViewND::<f32>::new(&mut clone) {
            Ok(view) => view,
            Err(e) => {
                error!("Failed to create MatView: {}", e);
                return Err(e);
            }
        };
        debug!("MatView loaded successfully");

        let mut valid_detections = 0;
        for i in 0..num {
            // if let Ok(confidence) = detections.at_nd::<f32>(&[0, 0, i as i32, 2]) {
            if let Ok(confidence) = detections.at_nd::<f32>(&[0, 0, i as i32, 2]) {
                if *confidence > self.confidence {
                    valid_detections += 1;
                    let class_id = match detections.at_nd::<f32>(&[0, 0, i as i32, 1]) {
                        Ok(id) => *id,
                        Err(e) => {
                            warning!("Failed to read class ID: {}", e);
                            continue;
                        }
                    };

                    if class_id >= Net::CLASSES.len() as f32 {
                        warning!(
                            "Invalid class ID: {}, max allowed: {}",
                            class_id,
                            Net::CLASSES.len() - 1
                        );
                        continue;
                    }

                    let class_name = Net::CLASSES[class_id as usize];
                    debug!(
                        "Detection #{}: class='{}', confidence={:.2}%",
                        i,
                        class_name,
                        *confidence * 100.0
                    );

                    if class_name != "person" {
                        debug!("Skipping detection for non-person class '{}'", class_name);
                        continue;
                    }

                    // let start_x = Net::mat_pos_default(&detect_clone, &[0, 0, i as i32, 3]) as i32;
                    let start_x = *mv.get(&[0, 0, i as i32, 3]).unwrap() as i32;
                    let start_y = *mv.get(&[0, 0, i as i32, 4]).unwrap() as i32;
                    let end_x = *mv.get(&[0, 0, i as i32, 5]).unwrap() as i32;
                    let end_y = *mv.get(&[0, 0, i as i32, 6]).unwrap() as i32;

                    debug!(
                        "Person detected at position: x=[{}, {}], y=[{}, {}]",
                        start_x, end_x, start_y, end_y
                    );

                    let rect = Rect::new(
                        start_x.max(0),
                        start_y.max(0),
                        (end_x - start_x).max(1),
                        (end_y - start_y).max(1),
                    );
                    debug!("Person rectangle: {:?}", rect);

                    self.detections
                        .lock()
                        .unwrap()
                        .push(Detection::new(rect, *confidence));

                    for (idx, point) in mv.iter().enumerate().take(5) {
                        // Only log first 5 points to avoid spam
                        debug!("Point {}: {:?}", idx, point);
                    }
                }
            }
        }

        info!(
            "Frame processing complete: found {} valid detections in {:?}",
            valid_detections,
            process_start.elapsed()
        );

        Ok(())
    }

    #[inline]
    fn mat_pos_default(mat: &Mat, pos: &[i32]) -> f32 {
        match mat.at_nd::<f32>(pos) {
            Ok(ok) => *ok,
            Err(_err) => {
                debug!(
                    "Failed to get mat position at {:?}, returning default 0.0",
                    pos
                );
                0.
            }
        }
    }

    fn create_tracker(&mut self, frame: &Mat, rect: Rect) -> Result<()> {
        let mut tracker = TrackerKCF::create(TrackerKCF_Params::default()?)?;

        tracker.init(frame, rect)?;

        self.trackers.push(Arc::new(Mutex::new(tracker)));
        Ok(())
    }

    fn update_trackers(&mut self, frame: &Mat) -> Result<()> {
        let mut valid_trackers: Vec<Arc<Mutex<Ptr<TrackerKCF>>>> = Vec::new();
        let mut current_rects = Vec::new();

        let mut temp_trackers = std::mem::take(&mut self.trackers);

		// NEXT Nice place to add parallel processing		

        for tracker in temp_trackers.drain(..) {
            let (success, bbox) = {
                let mut locked = match tracker.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        debug!("Poisoned tracker");
                        poisoned.into_inner() //  MAYBE Check if we should recover if it's poisoned
                    }
                };

				let mut bbox = Rect::default();
				let success = locked.update(&frame, &mut bbox)?;
				(success, bbox)
            };

			if success {
				valid_trackers.push(tracker);
				current_rects.push(bbox);
			}
        }

        self.trackers = valid_trackers;
        self.tracked_rects = current_rects;
        Ok(())
    }

	fn draw_tracking_results(&self, frame: &mut Mat) -> Result<()> {
		for rect in &self.tracked_rects {
			imgproc::rectangle(frame, *rect, Scalar::new(0., 255., 0., 0.), 2, imgproc::LINE_8, 0)?;
		}
		Ok(())	
	}
}

#[derive(Debug)]
pub struct Detection {
    pub detection: Rect,
    pub tod: Instant,
    pub confidence: f32,
}

impl Detection {
    pub fn new(detection: Rect, confidence: f32) -> Detection {
        Detection {
            detection,
            tod: Instant::now(),
            confidence,
        }
    }
}
