use anyhow::Result;
use log::{error, info};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{File, read_to_string},
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{fs, sync::mpsc, time::sleep};

pub struct TagDetection {
    tag: String,
    ant: i32,
    pot: i32,
    time: Instant,
}

struct TagDetections {
    pub detections: Arc<RwLock<VecDeque<TagDetection>>>,
}

impl TagDetection {
    fn new(tag: String, ant: i32, pot: i32, time: Instant) -> Self {
        Self {
            tag,
            ant,
            pot,
            time,
        }
    }

    fn new_now(tag: String, ant: i32, pot: i32) -> Self {
        Self {
            tag,
            ant,
            pot,
            time: Instant::now(),
        }
    }
}

impl TagDetections {
    pub fn new() -> Self {
        Self {
            detections: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
}

async fn process_spool(
    file: PathBuf,
    tag_re: &Regex,
    target_tag: &str,
    cv_tx: mpsc::Sender<TagDetection>,
    rate: f64,
) {
    let line_delay = Duration::from_secs_f64(1.0 / rate);

    loop {
        let content = match fs::read_to_string(&file).await {
            Ok(c) => c,
            Err(e) => {
                error!("Error reading file: {}", e);
                continue;
            }
        };

        for line in content.lines() {
            let Some((tag, rest)) = line.split_once(",") else {
                continue;
            };

            if !tag_re.is_match(tag) || !tag.contains(target_tag) {
                continue;
            }

            let parts: Vec<&str> = rest.split(',').collect();

            let Ok(ant) = parts[0].parse::<i32>() else {
                continue;
            };
            let Ok(rssi) = parts[1].trim_start_matches('-').parse::<i32>() else {
                continue;
            };

            if let Err(e) = cv_tx
                .send(TagDetection::new_now(tag.into(), ant, rssi))
                .await
            {
                info!("ALERTA ALERTA ALERTA {}", e);
                continue;
            };
        }

        if let Err(e) = fs::write(&file, "").await {
            error!("Error clearing file: {}. Marcianito Detected", e);
        }

        sleep(line_delay).await;
    }
}


