#!/bin/sh
# Compile a small native executable so validation does not launch the GUI or alter its dataset.
set -eu
cd "$(dirname "$0")/.."
bench_dir=$(mktemp -d "${TMPDIR:-/tmp}/subvox-benchmark.XXXXXX")
trap 'rm -rf "$bench_dir"' EXIT HUP INT TERM
xcrun swiftc -O -parse-as-library -enable-bare-slash-regex \
  Subvox/Motion/MotionSample.swift \
  Subvox/Decoding/*.swift \
  Subvox/Training/Experiment.swift Subvox/Training/TakeQuality.swift \
  Subvox/Training/TrainingSession.swift Subvox/Training/Phonemes.swift \
  Subvox/Training/PhoneEvaluation.swift Subvox/Training/DecoderEvaluation.swift \
  Subvox/Training/SignalCheck.swift \
  Subvox/Reading/Lexicon.swift Subvox/Reading/Alignment.swift \
  Tools/benchmark_learning.swift -o "$bench_dir/benchmark"
"$bench_dir/benchmark" "$@"
