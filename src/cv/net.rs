use crate::cv::centroid::CentroidTracker;
use crate::cv::mat_view::MatViewND;
use crate::direction::Direction;
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
use rayon::prelude::*;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::*;

#[derive(Debug, Clone)]
pub struct Net {
    net: dnn::Net,
    trackers: Vec<Arc<Mutex<Ptr<TrackerKCF>>>>,
    confidence: f32,
    tracked_rects: Vec<Rect>,
    skip_frames: u32,
    input_size: Size,
    frame_count: u32,
    centroid_tracker: CentroidTracker,
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
            confidence,
            tracked_rects: Vec::new(),
            skip_frames,
            input_size,
            frame_count: 0,
            centroid_tracker: CentroidTracker::new(3, 20.),
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
        #[cfg(debug_assertions)]
        debug!("Preprocessing frame: starting transformation pipeline");

        #[cfg(debug_assertions)]
        let start_time = Instant::now();

        let mut rotated = Mat::default();
        #[cfg(debug_assertions)]
        debug!("Rotating frame by 180 degrees");

        if let Err(e) = rotate(&frame, &mut rotated, ROTATE_180) {
            error!("Failed to rotate frame: {}", e);
            return Err(e);
        }

        let mut resized = Mat::default();
        #[cfg(debug_assertions)]
        debug!("Resizing frame to 500x500");

        match imgproc::resize(
            &frame,
            // &rotated, // Not rotating for testing
            &mut resized,
            self.input_size,
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

        #[cfg(debug_assertions)]
        debug!(
            "Frame preprocessing completed in {:?}",
            start_time.elapsed()
        );
        Ok(resized)
    }

    pub fn process_frame(&mut self, full_frame: &Mat) -> Result<Mat> {
        // 1. Run detection/tracking on a downscaled copy
        let small_size = self.input_size;
        let mut small = Mat::default();
        imgproc::resize(
            full_frame,
            &mut small,
            small_size,
            0.,
            0.,
            imgproc::INTER_AREA,
        )?;

        // 2. Detection or tracking on `small`
        if self.frame_count % self.skip_frames == 0 {
            self.trackers.clear();
            self.tracked_rects.clear();
            let detections = self.detect_objects(&small)?;
            for (rect, conf) in detections {
                if conf > self.confidence {
                    self.create_tracker(&small, rect)?;
                }
            }
        } else {
            self.update_trackers(&small)?;
        }

        let mid_y = 250; // TODO: Set the zones through config

        let rects = self.tracked_rects.clone();
        let objects = self.centroid_tracker.update(&rects)?;

        for (object_id, centroid) in &objects {
            if let Some(obj) = self.centroid_tracker.objects.get_mut(object_id) {
                if obj.centroids.len() >= 2 {
                    let prev_y = obj.centroids[obj.centroids.len() - 2].y;
                    let current_y = centroid.y;

                    let direction = if current_y < prev_y {
                        Direction::Up
                    } else {
                        Direction::Down
                    };

                    if !obj.counted {
                        if direction == Direction::Up && current_y < mid_y {
                            obj.counted = true;
                            info!("Obj: {} entered", obj.oid);
                        } else if direction == Direction::Down && current_y > mid_y {
                            obj.counted = true;
                            info!("Obj: {} exited", obj.oid);
                        }
                    }

                    obj.last_direction = Some(direction);
                }

                if obj.centroids.len() > 50 {
                    obj.centroids.remove(0);
                }
            };
        }

        // 3. Prepare output image (clone full resolution)
        let mut out = full_frame.clone();

        // 4. Draw scaled tracking results
        self.draw_tracking_results(&mut out)?;
        self.frame_count += 1;

        Ok(out)
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
        let temp_trackers = std::mem::take(&mut self.trackers);

        // Parallel processing of tracker updates
        let results: Vec<(bool, Rect, Arc<Mutex<Ptr<TrackerKCF>>>)> = temp_trackers
            .into_par_iter()
            .map(|tracker| {
                let (success, bbox) = {
                    let mut locked = match tracker.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            #[cfg(debug_assertions)]
                            debug!("Poisoned tracker");
                            poisoned.into_inner()
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

                (success, bbox, tracker)
            })
            .collect();

        // Filter successful trackers
        let mut valid_trackers = Vec::new();
        let mut current_rects = Vec::new();

        for (success, bbox, tracker) in results {
            if success {
                #[cfg(debug_assertions)]
                debug!("Succeeded updating trackers");
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
        // Compute scale factors from small (input_size) to full frame
        let fx = frame.cols() as f32 / self.input_size.width as f32;
        let fy = frame.rows() as f32 / self.input_size.height as f32;

        imgproc::line(
            frame,
            Point::new(0, 200),
            Point::new(frame.cols(), 200),
            Scalar::new(0., 255., 0., 0.),
            2,
            imgproc::LINE_8,
            0,
        )?;

        for rect in &self.tracked_rects {
            debug!("Original rect (small coords): {:?}", rect);
            let x = (rect.x as f32 * fx).round() as i32;
            let y = (rect.y as f32 * fy).round() as i32;
            let w = (rect.width as f32 * fx).round() as i32;
            let h = (rect.height as f32 * fy).round() as i32;
            let scaled = Rect::new(x, y, w, h);
            debug!("Drawing scaled rect: {:?}", scaled);
            imgproc::rectangle(
                frame,
                scaled,
                Scalar::new(0., 255., 0., 0.),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }
        Ok(())
    }

    fn detect_objects(&mut self, frame: &Mat) -> Result<Vec<(Rect, f32)>> {
        let mut detections: Vec<(Rect, f32)> = Vec::with_capacity(100); // Pre-allocate with reasonable capacity

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
                "Unexpected output size. Expected: 4 Received: {}",
                sizes.len()
            );
        }

        let num_detections = sizes[2] as usize;
        let mut clone = output.clone();
        let mv = MatViewND::<f32>::new(&mut clone)?;
        #[cfg(debug_assertions)]
        debug!("{:?} Raw net detections", 0..num_detections);

        // Parallel processing of detections
        let detections: Vec<(Rect, f32)> = (0..num_detections)
            .into_par_iter()
            .filter_map(|i| {
                // Get confidence and class_id safely
                let confidence = mv.get(&[0, 0, i as i32, 2]).ok()?;
                if *confidence <= self.confidence {
                    return None;
                }

                let class_id = mv.get(&[0, 0, i as i32, 1]).ok()?;

                // Check for person class
                if (*class_id as usize) < Self::CLASSES.len()
                    && Self::CLASSES[*class_id as usize] == "person"
                {
                    // Calculate bounding box inline for better performance
                    let (w, h) = (self.input_size.width as f32, self.input_size.height as f32);
                    let start_x = mv.get(&[0, 0, i as i32, 3]).ok().map(|v| (*v * w) as i32)?;
                    let start_y = mv.get(&[0, 0, i as i32, 4]).ok().map(|v| (*v * h) as i32)?;
                    let end_x = mv.get(&[0, 0, i as i32, 5]).ok().map(|v| (*v * w) as i32)?;
                    let end_y = mv.get(&[0, 0, i as i32, 6]).ok().map(|v| (*v * h) as i32)?;

                    let rect = Rect::new(
                        start_x.max(0),
                        start_y.max(0),
                        (end_x - start_x).max(1),
                        (end_y - start_y).max(1),
                    );

                    Some((rect, *confidence))
                } else {
                    None
                }
            })
            .collect();

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
