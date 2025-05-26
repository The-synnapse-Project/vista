use anyhow::{Context, Result, anyhow};
use csv::{Writer, WriterBuilder};
use log::{debug, info};
use opencv::{
    core::{Mat, MatTraitConst, Size},
    videoio::{
        VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, VideoWriter, VideoWriterTrait,
    },
};
use std::{
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt},
    signal,
    sync::watch,
    task,
    time::interval,
};

#[derive(Debug, Clone)]
pub struct SyncronizedRecorderConfig {
    pub camera_path: PathBuf,
    pub rfid_path: PathBuf,
    pub output_video: PathBuf,
    pub output_video_timestamps: PathBuf,
    pub output_detections: PathBuf,
    pub duty_cycle: u64,
}

pub struct SyncronizedRecorder {
    config: SyncronizedRecorderConfig,
}

impl SyncronizedRecorder {
    pub fn new(config: SyncronizedRecorderConfig) -> Self {
        Self { config }
    }

    pub async fn start(self) -> Result<()> {
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        info!("Starting video recorder");
        let video_handle = task::spawn_blocking({
            let config = self.config.clone();
            let shutdown = shutdown_tx.clone();
            move || Self::video_task(config, shutdown)
        });

        info!("Starting rfid recorder");
        let serial_handle =
            task::spawn(Self::serial_task(self.config.clone(), shutdown_tx.clone()));

        let ctrl_c = async {
            signal::ctrl_c().await?;
            anyhow::Ok(())
        };

        tokio::select! {
            _ = ctrl_c => {
                shutdown_tx.send(true)?;
            }
            result = video_handle => {
                result??;
            }
            result = serial_handle => {
                result??;
            }
        }

        let _ = shutdown_rx.changed().await;

        Ok(())
    }

    fn video_task(config: SyncronizedRecorderConfig, shutdown: watch::Sender<bool>) -> Result<()> {
        let camera_path = config.camera_path.to_str().context("Invalid Camera Path")?;

        let mut camera =
            VideoCapture::from_file_def(camera_path).context("Failed to open camera")?;

        if !camera.is_opened()? {
            return Err(anyhow!("Camera not opened"));
        }

        let fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G')?;
        let mut writer = VideoWriter::new(
            config.output_video.to_str().context("Invalid video path")?,
            fourcc,
            30.,
            Size::new(640, 480),
            true,
        )?;

        let mut ts_writer = Writer::from_path(&config.output_video_timestamps)?;
        ts_writer.write_record(&["frame_number", "timestamp"])?;

        let mut frame = Mat::default();
        let mut frame_count = 0;

        info!("Video recorder initialized");
        loop {
            if *shutdown.borrow() {
                break;
            }

            camera.read(&mut frame)?;
            if frame.empty() {
                debug!("Empty frame: {}", frame_count);
                continue;
            }

            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

            writer.write(&frame)?;
            ts_writer.write_record(&[frame_count.to_string(), timestamp.to_string()])?;
            ts_writer.flush()?;
            debug!("Wrote frame data: {}", frame_count);
            frame_count += 1;
        }

        Ok(())
    }

    async fn serial_task(
        config: SyncronizedRecorderConfig,
        shutdown: watch::Sender<bool>,
    ) -> Result<()> {
        let mut det_writer = WriterBuilder::new()
            .quote_style(csv::QuoteStyle::Always)
            .from_path(&config.output_detections)?;
        det_writer.write_record(&["timestamp", "data"])?;

        let mut interval = interval(Duration::from_millis(config.duty_cycle));
        let mut buffer = String::new();

        info!("Initialized serial recorder");
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if *shutdown.borrow() {
                        break;
                    }

                    let mut file = OpenOptions::new().read(true).write(true).open(&config.rfid_path).await?;

                    file.read_to_string(&mut buffer).await?;
                    if !buffer.is_empty() {
                        // info!("SERIAL: Read {} lines", buffer.lines().max());
                        for line in buffer.lines() {
                            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

                            det_writer.write_record(&[timestamp.to_string(), line.trim().to_string()])?;
                        }
                        det_writer.flush()?;

                        file.seek(std::io::SeekFrom::Start(0)).await?;
                        file.set_len(0).await?;
                        buffer.clear();
                    }
                }
            }
        }

        Ok(())
    }
}
