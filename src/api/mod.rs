use anyhow::Result;
use reqwest::Response;
use serde::{Deserialize, Serialize};

use crate::auth::auth;

#[derive(Serialize, Deserialize)]
pub struct APIDetectionRequest {
    person_id: String,
    action: String,
}

#[derive(Serialize, Deserialize)]
pub struct APIDetectionResponse {
    person_id: String,
    action: String,
}

struct ApiSpec {
    base_url: String, // Changed to owned String
}

impl ApiSpec {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    pub fn post_url(&self) -> String {
        format!("{}/api/entry", self.base_url)
    }

    pub fn post_hmac(&self) -> String {
        auth::compute_hmac(&self.post_url())
    }

    pub fn put_url(&self, entry_id: String) -> String {
        format!("{}/api/entry/{}", self.base_url, entry_id)
    }

    pub fn delete_url(&self, entry_id: String) -> String {
        format!("{}/api/entry/{}", self.base_url, entry_id)
    }
}

struct Api {
    spec: ApiSpec,
    client: reqwest::Client,
}

impl Api {
    pub fn new(base_url: &str) -> Self {
        Self {
            spec: ApiSpec::new(base_url.to_owned()),
            client: reqwest::Client::new(),
        }
    }

    pub async fn add_detection(&self, detection: APIDetectionRequest) -> Result<()> {
        self.client
            .post(self.spec.post_url())
            .header("X-Syn-Api-Key", self.spec.post_hmac())
            .json(&detection)
            .send()
            .await?;

        Ok(())
    }

    // pub async fn change_detection() {

    // }
}
