use anyhow::{Context, Result, anyhow};
use csv::{Writer, WriterBuilder};
use log::{debug, info, warning};
use opencv::{
    core::{Mat, MatTraitConst, Size},
    videoio::{
        VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, VideoWriter, VideoWriterTrait,
    },
};
use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::OpenOptions,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    signal,
    sync::watch,
    task,
    time::interval,
};

#[derive(Debug, Clone)]
pub struct SynchronizedRecorderConfig {
    pub camera_path: PathBuf,
    pub rfid_path: PathBuf,
    pub read_lock: PathBuf,
    pub output_video: PathBuf,
    pub output_video_timestamps: PathBuf,
    pub output_detections: PathBuf,
    pub duty_cycle: u64,
}

pub struct SynchronizedRecorder {
    config: SynchronizedRecorderConfig,
}

impl SynchronizedRecorder {
    pub fn new(config: SynchronizedRecorderConfig) -> Self {
        Self { config }
    }

    pub async fn start(self) -> Result<()> {
        info!("Starting synchronized recorder");
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        info!("Spawning video recorder task");
        let video_handle = task::spawn_blocking({
            let config = self.config.clone();
            let shutdown = shutdown_tx.clone();
            move || Self::video_task(config, shutdown)
        });

        info!("Spawning RFID recorder task");
        let serial_handle =
            task::spawn(Self::serial_task(self.config.clone(), shutdown_tx.clone()));

        let ctrl_c = async {
            signal::ctrl_c().await?;
            info!("Received CTRL+C signal");
            anyhow::Ok(())
        };

        tokio::select! {
            _ = ctrl_c => {
                info!("Initiating shutdown procedure");
                shutdown_tx.send(true).context("Failed to send shutdown signal")?;
            }
            result = video_handle => {
                result?.context("Video task failed")?;
            }
            result = serial_handle => {
                result?.context("Serial task failed")?;
            }
        }

        info!("Waiting for tasks to complete");
        shutdown_rx.changed().await?;
        info!("All tasks completed");

        Ok(())
    }

    fn video_task(config: SynchronizedRecorderConfig, shutdown: watch::Sender<bool>) -> Result<()> {
        let camera_path = config.camera_path.to_str().context("Invalid Camera Path")?;
        info!("Opening camera at: {}", camera_path);

        let mut camera =
            VideoCapture::from_file_def(camera_path).context("Failed to open camera")?;

        if !camera.is_opened()? {
            return Err(anyhow!("Camera not opened"));
        }

        let output_video = config.output_video.to_str().context("Invalid video path")?;
        info!("Creating video writer for: {}", output_video);
        let fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G')?;
        let mut writer = VideoWriter::new(output_video, fourcc, 30., Size::new(640, 480), true)?;

        let ts_path = config.output_video_timestamps.to_string_lossy();
        info!("Creating timestamp CSV: {}", ts_path);
        let mut ts_writer = Writer::from_path(&config.output_video_timestamps)?;
        ts_writer.write_record(["frame_number", "timestamp"])?;

        let mut frame = Mat::default();
        let mut frame_count = 0;
        let mut last_log_frame = 0;
        let log_interval = 100; // Log every 100 frames

        info!("Starting video capture loop");
        while !*shutdown.borrow() {
            if !camera.read(&mut frame)? {
                warning!("Failed to read frame from camera");
                continue;
            }

            if frame.empty() {
                debug!("Empty frame received");
                continue;
            }

            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

            writer.write(&frame)?;
            ts_writer.write_record(&[frame_count.to_string(), timestamp.to_string()])?;

            if frame_count % log_interval == 0 {
                debug!(
                    "Written {} frames (last timestamp: {})",
                    frame_count, timestamp
                );
                ts_writer.flush()?;
            }

            frame_count += 1;

            // Periodic progress logging
            if frame_count - last_log_frame >= 1000 {
                info!("Processed {} frames", frame_count);
                last_log_frame = frame_count;
            }
        }

        ts_writer.flush()?;
        info!(
            "Video task completed. Total frames: {}, shutting down",
            frame_count
        );
        Ok(())
    }

    async fn serial_task(
        config: SynchronizedRecorderConfig,
        shutdown: watch::Sender<bool>,
    ) -> Result<()> {
        info!(
            "Starting RFID polling loop (duty cycle: {}ms)",
            config.duty_cycle
        );

        let det_path = config.output_detections.to_string_lossy();
        info!("Creating detections CSV: {}", det_path);
        let mut det_writer = WriterBuilder::new()
            .quote_style(csv::QuoteStyle::Always)
            .from_path(&config.output_detections)?;
        det_writer.write_record(["timestamp", "data"])?;

        let mut interval = interval(Duration::from_millis(config.duty_cycle));
        let mut buffer = String::new();

        while !*shutdown.borrow() {
            interval.tick().await;

            let read = tokio::fs::read_to_string(&config.read_lock)
                .await
                .unwrap_or_else(|_| "0".to_string());
            let read = read.trim();

            if read == "1" {
                debug!("Reading RFID");

                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&config.rfid_path)
                    .await
                    .context("Failed to read RFID spool")?;

                file.read_to_string(&mut buffer).await?;

                if !buffer.is_empty() {
                    debug!("Read {} bytes from RFID spool", buffer.len());
                    let line_count = buffer.lines().count();

                    for line in buffer.lines() {
                        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
                        det_writer
                            .write_record(&[timestamp.to_string(), line.trim().to_string()])?;
                    }
                    det_writer.flush()?;
                    info!("Wrote {} entries to CSV", line_count);

                    file.set_len(0).await?;
                    debug!("Errased RFID file");
                    buffer.clear();
                }
            } else {
                debug!("Clearing spool (not reading)");

                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&config.rfid_path)
                    .await?;

                file.write_all(b"").await?;
            }
        }

        det_writer.flush()?;
        info!("RFID task completed, shutting down");
        Ok(())
    }
}
