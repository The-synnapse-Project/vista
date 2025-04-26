use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Conf {
    pub version: u8,
    pub db_conn: String,
}

impl ::std::default::Default for Conf {
    fn default() -> Self {
        Self {
            version: 0,
            db_conn: "mysql://username:password@host/database".into(),
        }
    }
}

pub fn load_config() -> Result<Conf, confy::ConfyError> {
    confy::load("vista", None)
}
