use hex::encode;
use hmac::{Hmac, Mac};
use once_cell::sync::Lazy;
use sha2::Sha256;
use std::env::var;

static API_SECRET: Lazy<String> =
    Lazy::new(|| var("SYN_API_SECRET").unwrap_or_else(|_| "secret".to_string()));

pub fn verify_api_key(api_key: &str, uri: &str) -> bool {
    let expected_hmac = compute_hmac(uri);

    expected_hmac == api_key
}

pub fn compute_hmac(uri: &str) -> String {
    type HmacSha256 = Hmac<Sha256>;

    let mut mac =
        HmacSha256::new_from_slice(API_SECRET.as_bytes()).expect("HMAC can take key of any size");
    mac.update(uri.as_bytes());

    let result = mac.finalize();
    let code_bytes = result.into_bytes();

    encode(code_bytes)
}
