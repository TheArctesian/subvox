#!/bin/sh
# Audit development recordings locally without launching the GUI or changing the dataset.
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
  Tools/audit_signal.swift -o "$bench_dir/benchmark"
"$bench_dir/benchmark" "$@"
