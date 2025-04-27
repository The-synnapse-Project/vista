use log::{debug, info};
use std::time::{Duration, Instant};

pub struct FrameMetrics {
    last_frame_time: Instant,
    fps: f32,
    frame_count: usize,
    avg_fps: f32,
    min_fps: f32,
    max_fps: f32,
    start_time: Instant,
}

impl FrameMetrics {
    pub fn new() -> Self {
        debug!("Initializing frame metrics tracker");
        FrameMetrics {
            last_frame_time: Instant::now(),
            fps: 0.0,
            frame_count: 0,
            avg_fps: 0.0,
            min_fps: f32::MAX,
            max_fps: 0.0,
            start_time: Instant::now(),
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame_time);
        let current_fps = 1.0 / elapsed.as_secs_f32();

        self.fps = current_fps;
        self.frame_count += 1;

        // Update statistics
        self.min_fps = self.min_fps.min(current_fps);
        self.max_fps = self.max_fps.max(current_fps);

        // Recalculate average FPS based on total runtime
        let total_runtime = self.start_time.elapsed().as_secs_f32();
        self.avg_fps = self.frame_count as f32 / total_runtime;

        // Log performance data periodically (every 100 frames)
        if self.frame_count % 100 == 0 {
            info!(
                "Performance stats after {} frames: Current: {:.1} FPS, Avg: {:.1} FPS, Min: {:.1} FPS, Max: {:.1} FPS",
                self.frame_count, self.fps, self.avg_fps, self.min_fps, self.max_fps
            );
        } else {
            debug!(
                "Frame #{}: {:.1} FPS (frame time: {:.1}ms)",
                self.frame_count,
                self.fps,
                elapsed.as_millis()
            );
        }

        self.last_frame_time = now;
    }

    pub fn get_fps(&self) -> f32 {
        self.fps
    }

    pub fn get_last_frame_time(&self) -> Duration {
        self.last_frame_time.elapsed()
    }

    pub fn get_avg_fps(&self) -> f32 {
        self.avg_fps
    }

    pub fn get_min_fps(&self) -> f32 {
        self.min_fps
    }

    pub fn get_max_fps(&self) -> f32 {
        self.max_fps
    }

    pub fn get_frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn get_total_runtime(&self) -> Duration {
        self.start_time.elapsed()
    }
}
