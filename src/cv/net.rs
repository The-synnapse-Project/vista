use log::info;
use opencv::core::*;
use opencv::dnn;
use opencv::dnn::NetTrait;
use opencv::imgproc;
use std::os::raw::c_double;
use std::sync::Arc;
use std::time::*;
use tokio::{sync::mpsc, task};

pub struct Net {
    net: dnn::Net,
    detections: Arc<Vec<Detection>>,
    confidence: f32,
}

impl Net {
    pub fn new(prototxt: &str, caffe_model: &str) -> opencv::Result<Self> {
        let net = dnn::read_net_from_caffe(prototxt, caffe_model)?;
        Ok(Self {
            net,
            detections: Arc::new(Vec::new()),
            confidence: 0.,
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

        Ok(grayscale)
    }

    pub fn process_frame(mut self, frame: &Mat) {
        let frame_blob = dnn::blob_from_image(
            &frame,
            1.0 / 127.5,
            Size::new(500, 500),
            Scalar::new(127.5, 127.5, 127.5, 0.),
            false,
            false,
            CV_32F,
        );

        // MAYBE Nice place to start doing async with fordward_async_def

        if let Ok(frame_blob) = frame_blob {
            self.net.set_input_def(&frame_blob);
            if let Ok(detections) = self.net.forward_single_def() {
                let sizes = detections.mat_size();

                if sizes.len() == 4 {
                    let num = sizes[2] as usize;

                    for i in 0..num {
                        if let Ok(confidence) = detections.at_nd::<f32>(&[0, 0, i as i32, 2]) {
                            if *confidence > self.confidence {
                                if let Ok(class_id) = detections.at_nd::<f32>(&[0, 0, i as i32, 1])
                                {
                                    if *class_id < Net::CLASSES.len() as f32
                                        && Net::CLASSES[*class_id as usize] == "person"
                                    {
                                        let start_x =
                                            Net::mat_pos_default(&detections, &[0, 0, i as i32, 3])
                                                as i32;
                                        let start_y =
                                            Net::mat_pos_default(&detections, &[0, 0, i as i32, 4])
                                                as i32;
                                        let end_x =
                                            Net::mat_pos_default(&detections, &[0, 0, i as i32, 5])
                                                as i32;
                                        let end_y =
                                            Net::mat_pos_default(&detections, &[0, 0, i as i32, 6])
                                                as i32;

                                        let rect = Rect::new(
                                            start_x.max(0),
                                            start_y.max(0),
                                            (end_x - start_x).max(1),
                                            (end_y - start_y).max(1),
                                        );

                                        info!("Recta {rect:?}");
                                    }
                                };
                            }
                        }
                    }
                }
            }
        };
    }

    #[inline]
    fn mat_pos_default(mat: &Mat, pos: &[i32]) -> f32 {
        match mat.at_nd::<f32>(pos) {
            Ok(ok) => *ok,
            Err(err) => 0.,
        }
    }
}

#[derive(Debug)]
pub struct Detection {
    pub detection: Rect,
    pub tod: Instant,
}

impl Detection {
    pub fn new(detection: Rect) -> Detection {
        Detection {
            detection,
            tod: Instant::now(),
        }
    }
}
