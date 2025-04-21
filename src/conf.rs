use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize)]
struct Conf {
	version: u8,
	db_conn: String
}

impl ::std::default::Default for Conf {
	fn default() -> Self {
		Self { version: 0, db_conn: "mysql://username:password@host/database".into() }
	}
}

pub fn load_config() -> Result<(), Box<dyn std::error::Error>> {
	let cfg: Conf = confy::load("vista", None)?;
	dbg!(cfg);
	Ok(())
}
