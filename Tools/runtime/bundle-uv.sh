#!/bin/sh
# Puts a pinned, checksum-verified uv into the app at Contents/Helpers/uv, so end users never
# install anything: the app's Python runtime (SequenceRuntime) runs through this copy.
#
# Runs as an Xcode build phase. Downloads happen once per machine into a cache; a missing or
# mismatched checksum fails the build rather than shipping an unverified binary.
#   bundle-uv.sh <destination-file>
# Uses ARCHS and EXPANDED_CODE_SIGN_IDENTITY from Xcode when set.
set -eu

VERSION=0.13.6
SHA_ARM64=60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472
SHA_X86_64=63ba17fdda55ad1a230d4b37fc6017ffa10c23688c83818bd3bfdf914c652a02

destination=${1:?usage: bundle-uv.sh <destination-file>}
cache="${SUBVOX_UV_CACHE:-$HOME/Library/Caches/Subvox-build/uv}/$VERSION"
mkdir -p "$cache"

fetch() { # <rust-target> <sha256> -> path of the verified uv binary for that target
    archive="$cache/uv-$1.tar.gz"
    if [ ! -f "$archive" ] || [ "$(shasum -a 256 "$archive" | cut -d' ' -f1)" != "$2" ]; then
        curl --fail --silent --show-error --location --retry 3 \
            "https://github.com/astral-sh/uv/releases/download/$VERSION/uv-$1.tar.gz" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    actual=$(shasum -a 256 "$archive" | cut -d' ' -f1)
    if [ "$actual" != "$2" ]; then
        rm -f "$archive"
        echo "error: uv $VERSION for $1 has checksum $actual, expected $2" >&2
        exit 1
    fi
    tar -xzf "$archive" -C "$cache" "uv-$1/uv"
    echo "$cache/uv-$1/uv"
}

binaries=""
for arch in ${ARCHS:-arm64 x86_64}; do
    case $arch in
        arm64) binaries="$binaries $(fetch aarch64-apple-darwin "$SHA_ARM64")" ;;
        x86_64) binaries="$binaries $(fetch x86_64-apple-darwin "$SHA_X86_64")" ;;
        *) echo "error: no uv build for architecture $arch" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$destination")"
rm -f "$destination"
# shellcheck disable=SC2086 # one path per architecture
lipo -create $binaries -output "$destination"
chmod 755 "$destination"
# Nested code must be signed before Xcode signs the app around it; ad hoc when no identity is set.
codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY:--}" --timestamp=none "$destination"
"$destination" --version
