use std::time::{Duration, Instant};

pub struct FrameMetrics {
    last_frame_time: Instant,
    fps: f32,
}

impl FrameMetrics {
    pub fn new() -> Self {
        FrameMetrics {
            last_frame_time: Instant::now(),
            fps: 0.0,
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame_time);
        self.fps = 1.0 / elapsed.as_secs_f32();
        self.last_frame_time = now;
    }

    pub fn get_fps(&self) -> f32 {
        self.fps
    }

    pub fn reset(&mut self) {
        self.last_frame_time = Instant::now();
        self.fps = 0.0;
    }
    pub fn get_last_frame_time(&self) -> Duration {
        self.last_frame_time.elapsed()
    }
}
