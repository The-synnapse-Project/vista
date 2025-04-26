use clap::Parser;
use cli::{parse_args, Args};
use conf::load_config;

mod cli;
mod db;
mod conf;
mod cv;

fn main() {
	let args: Args = parse_args();
	let cfg = load_config();
}
