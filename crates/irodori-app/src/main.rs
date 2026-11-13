use clap::Parser;
use irodori_app::launch::{run, Args};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run(Args::parse()).await
}
