use crate::cv::mat_view::MatViewND;
use anyhow::{Result, bail};
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
    input_size: Size,
    frame_count: u32,
}

impl Net {
    pub fn new(
        prototxt: &str,
        caffe_model: &str,
        confidence: f32,
        skip_frames: u32,
        input_size: Size,
    ) -> opencv::Result<Self> {
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
            confidence,
            tracked_rects: Vec::new(),
            skip_frames,
            input_size,
            frame_count: 0,
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

    pub fn preprocess_frame(&self, frame: &Mat) -> opencv::Result<Mat> {
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

        debug!(
            "Frame preprocessing completed in {:?}",
            start_time.elapsed()
        );
        Ok(resized)
    }

    pub fn process_frame(&mut self, frame: &Mat) -> Result<Mat> {
        let mut rgb = self.preprocess_frame(&frame)?;

        debug!(
            "Frame: {}, Skip: {}, Curr: {}",
            self.frame_count,
            self.skip_frames,
            self.frame_count % self.skip_frames
        );
        if self.frame_count % self.skip_frames == 0 {
            self.trackers.clear();
            self.tracked_rects.clear();

            let detections = self.detect_objects(&rgb)?;
            debug!("Tracking frame, detections: {}", detections.len());
            for (rect, confidence) in detections {
                if confidence > self.confidence {
                    self.create_tracker(&rgb, rect)?;
                }
            }

            debug!("Created {} trackers", self.trackers.len());
        } else {
            self.update_trackers(&rgb)?;
            debug!("Updating trackers");
        }
		
        self.draw_tracking_results(&mut rgb)?;
        self.frame_count += 1;
        Ok(rgb)
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

        tracker
            .init(frame, rect)
            .map_err(|e| Error::new(1, format!("Tracker init failed: {}", e)))?;

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
                let success = match locked.update(&frame, &mut bbox) {
                    Ok(s) => s,
                    Err(e) => {
                        error!("Tracker update failed: {}", e);
                        false
                    }
                };
                (success, bbox)
            };

            if success {
                debug!("Succeded updating trackers");
                valid_trackers.push(tracker);
                current_rects.push(bbox);
            }
        }

        self.trackers = valid_trackers;
        self.tracked_rects = current_rects;
        Ok(())
    }

    pub fn draw_tracking_results(&self, frame: &mut Mat) -> Result<()> {
        debug!("drawing {:?} recs", self.tracked_rects);
        for rect in &self.tracked_rects {
            debug!("Drawing rect {:?}", rect);
            imgproc::rectangle(
                frame,
                *rect,
                Scalar::new(0., 255., 0., 0.),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }
        Ok(())
    }

    fn detect_objects(&mut self, frame: &Mat) -> Result<Vec<(Rect, f32)>> {
        let mut detections = Vec::new();

        let blob = dnn::blob_from_image(
            frame,
            1. / 127.5,
            self.input_size,
            Scalar::new(127.5, 127.5, 127.5, 0.),
            true,
            false,
            CV_32F,
        )?;

        self.net.set_input_def(&blob);

        // TODO: Find out if this is the right function to use
        let output = self.net.forward_single("detection_out")?;
        debug!("Ran net fwd");

        let sizes = output.mat_size();
        if sizes.len() != 4 {
            bail!(
                "Unexpected output size. Expected: 4 Revived: {}",
                sizes.len()
            );
        }

        let num_detections = sizes[2] as usize;
        let mut clone = output.clone();
        let mv = MatViewND::<f32>::new(&mut clone)?;
        debug!("{:?} Raw net detections", 0..num_detections);

        for i in 0..num_detections {
            let confidence = mv.get(&[0, 0, i as i32, 2])?;
            if *confidence > self.confidence {
                let class_id = mv.get(&[0, 0, i as i32, 1])?;
                debug!(
                    "Class id: {}, Confidence: {}, Confidence threshold: {}",
                    *class_id, *confidence, self.confidence
                );
                debug!("Len: {}", Self::CLASSES.len() as f32);
                if *class_id <= Self::CLASSES.len() as f32
                    && Self::CLASSES[*class_id as usize] == "person"
                {
                    let rect = self.get_bounding_box(&mv, i)?;
                    debug!(
                        "Found {}, Confidence: {}, Rect: {:?}",
                        Self::CLASSES[*class_id as usize],
                        *confidence,
                        rect
                    );
                    detections.push((rect, *confidence));
                }
            }
        }

        Ok(detections)
    }

    fn get_bounding_box(&self, mv: &MatViewND<f32>, idx: usize) -> Result<Rect> {
        let (w, h) = (self.input_size.width as f32, self.input_size.height as f32);
        let start_x = (mv.get(&[0, 0, idx as i32, 3])? * w) as i32;
        let start_y = (mv.get(&[0, 0, idx as i32, 4])? * h) as i32;
        let end_x = (mv.get(&[0, 0, idx as i32, 5])? * w) as i32;
        let end_y = (mv.get(&[0, 0, idx as i32, 6])? * h) as i32;

        Ok(Rect::new(
            start_x.max(0),
            start_y.max(0),
            (end_x - start_x).max(1),
            (end_y - start_y).max(1),
        ))
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
