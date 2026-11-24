#!/bin/bash
# Regenerates the compressed logs of tests/data from their inputs.
#
#   tests/data/regenerate.sh gaussian/h2co orca6/functionals ...
#
# Each folder has a jobs.txt file: one job per line, in order, with for Gaussian
# the optional job whose checkpoint is copied first ("FREQ OPT": FREQ starts from OPT.chk).
# The jobs run in a temporary folder; only the gzipped logs are copied back.
# Programs: Gaussian 09 (g09root, default /applis) and ORCA (ORCA_BIN, default
# /applis/orca6.1/bin/orca). gaussian/h2o comes from examples/ and orca5/ from ORCA 5.0.4:
# they have no jobs.txt and are not regenerated.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ORCA_BIN=${ORCA_BIN:-/applis/orca6.1/bin/orca}
export g09root=${g09root:-/applis}

for dir in "$@"; do
    src="$HERE/$dir"
    [ -f "$src/jobs.txt" ] || { echo "$dir: no jobs.txt, skipped"; continue; }
    work=$(mktemp -d)
    cp "$src"/*.inp "$work"/
    ls "$src"/*.xyz > /dev/null 2>&1 && cp "$src"/*.xyz "$work"/
    case $dir in
        gaussian/*)
            set +e   # g09.profile ends with a non-zero status
            source "$g09root/g09/bsd/g09.profile"
            set -e
            export GAUSS_SCRDIR=$work
            grep -v '^#' "$src/jobs.txt" | while read -r job from; do
                [ -n "$from" ] && cp "$work/$from.chk" "$work/$job.chk"
                (cd "$work" && g09 "$job.inp")
                grep -q "Normal termination" "$work/$job.log" || echo "WARNING: $dir/$job did not end normally"
                gzip -9nc "$work/$job.log" > "$src/$job.log.gz"
            done ;;
        orca*)
            grep -v '^#' "$src/jobs.txt" | while read -r job; do
                (cd "$work" && "$ORCA_BIN" "$job.inp" > "$job.out" 2>&1) || true
                grep -q "ORCA TERMINATED NORMALLY" "$work/$job.out" || echo "WARNING: $dir/$job did not end normally"
                gzip -9nc "$work/$job.out" > "$src/$job.out.gz"
            done ;;
    esac
    rm -rf "$work"
    echo "$dir: done"
done
