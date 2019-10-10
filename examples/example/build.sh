#!/bin/sh

set -ex

rustfmt src/lib.rs
wasm-pack build --release --target web
rm pkg/.gitignore   # I need the code for GitHub Pages
python3 -m http.server