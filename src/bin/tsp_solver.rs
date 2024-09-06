use clap::Parser;
#[derive(Parser, Default, Debug)]
struct Arguments {
    package_name: String,
    max_depth: usize,
}

fn main() {
    let args = Arguments::parse();
    println!("{:?}", args);
}