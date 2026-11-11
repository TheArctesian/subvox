#!/bin/sh
# Puts a pinned whisper.cpp whisper-server into the app at Contents/Helpers/whisper-server, so
# local transcription needs nothing installed outside Subvox (LocalWhisper runs this copy).
#
# whisper.cpp publishes no macOS server binary, so this builds one from the checksum-verified
# source release with a checksum-verified CMake download: the build machine needs only Xcode.
# Each architecture is built once per machine into a cache, statically linked (Metal shaders
# embedded, Accelerate from the system), then merged into one universal binary and signed.
#   bundle-whisper-server.sh <destination-file>
# Uses ARCHS and EXPANDED_CODE_SIGN_IDENTITY from Xcode when set.
set -eu

VERSION=1.9.5
SOURCE_SHA=ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720
CMAKE_VERSION=4.4.5
CMAKE_SHA=679e77a96b9ef9484cc6c1445255fc4f84206b48a65420efbedf16bdab55fbff
DEPLOYMENT_TARGET=14.0

destination=${1:?usage: bundle-whisper-server.sh <destination-file>}
cache="${SUBVOX_WHISPER_CACHE:-$HOME/Library/Caches/Subvox-build/whisper.cpp}"
mkdir -p "$cache"

download() { # <url> <sha256> <file>
    if [ ! -f "$3" ] || [ "$(shasum -a 256 "$3" | cut -d' ' -f1)" != "$2" ]; then
        curl --fail --silent --show-error --location --retry 3 "$1" -o "$3.part"
        mv "$3.part" "$3"
    fi
    actual=$(shasum -a 256 "$3" | cut -d' ' -f1)
    if [ "$actual" != "$2" ]; then
        rm -f "$3"
        echo "error: $1 has checksum $actual, expected $2" >&2
        exit 1
    fi
}

build() { # <arch> -> path of the built whisper-server for that architecture
    output="$cache/$VERSION/$1/whisper-server"
    if [ -x "$output" ]; then echo "$output"; return; fi

    cmake="$cache/cmake-$CMAKE_VERSION/CMake.app/Contents/bin/cmake"
    if [ ! -x "$cmake" ]; then
        download "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-macos-universal.tar.gz" \
            "$CMAKE_SHA" "$cache/cmake-$CMAKE_VERSION.tar.gz"
        rm -rf "$cache/cmake-$CMAKE_VERSION"
        tar -xzf "$cache/cmake-$CMAKE_VERSION.tar.gz" -C "$cache"
        mv "$cache/cmake-$CMAKE_VERSION-macos-universal" "$cache/cmake-$CMAKE_VERSION"
    fi

    source="$cache/$VERSION/whisper.cpp-$VERSION"
    if [ ! -f "$source/CMakeLists.txt" ]; then
        download "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v$VERSION.tar.gz" \
            "$SOURCE_SHA" "$cache/whisper.cpp-$VERSION.tar.gz"
        mkdir -p "$cache/$VERSION"
        tar -xzf "$cache/whisper.cpp-$VERSION.tar.gz" -C "$cache/$VERSION"
    fi

    # x86_64: macOS 14 runs only on Intel Macs with AVX2/FMA/F16C, so those are safe to require.
    case $1 in
        arm64) cpu="" ;;
        x86_64) cpu="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" ;;
    esac
    work="$cache/$VERSION/build-$1"
    rm -rf "$work"
    # Xcode's build settings would otherwise leak into the nested compiler invocations.
    # shellcheck disable=SC2086 # cpu holds separate flags
    env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$HOME" DEVELOPER_DIR="${DEVELOPER_DIR:-}" \
        "$cmake" -S "$source" -B "$work" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="$1" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
        -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_METAL_EMBED_LIBRARY=ON $cpu \
        -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_SDL2=OFF -DWHISPER_CURL=OFF -DWHISPER_COREML=OFF >&2
    env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$HOME" DEVELOPER_DIR="${DEVELOPER_DIR:-}" \
        "$cmake" --build "$work" --config Release --target whisper-server -j "$(sysctl -n hw.ncpu)" >&2
    mkdir -p "$(dirname "$output")"
    cp "$work/bin/whisper-server" "$output.part"
    mv "$output.part" "$output"
    rm -rf "$work"
    echo "$output"
}

binaries=""
for arch in ${ARCHS:-arm64 x86_64}; do
    case $arch in
        arm64 | x86_64) binaries="$binaries $(build "$arch")" ;;
        *) echo "error: no whisper-server build for architecture $arch" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$destination")"
rm -f "$destination"
# shellcheck disable=SC2086 # one path per architecture
lipo -create $binaries -output "$destination"
chmod 755 "$destination"
# Statically linked: only system libraries may remain, so the helper runs on any Mac.
if otool -L "$destination" | awk '/^[[:space:]]/ && $1 !~ /^\/(usr\/lib|System\/Library)\// { found = 1 } END { exit !found }'; then
    otool -L "$destination" >&2
    echo "error: whisper-server links a non-system library" >&2
    exit 1
fi
# Nested code must be signed before Xcode signs the app around it; ad hoc when no identity is set.
codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY:--}" --timestamp=none "$destination"
