#!/usr/bin/env bash
# Fails when a tracked file names ArmarX, the ARMAR robots, Axii, H2T or KIT, or an internal path.
#
#     tools/check_references.sh
#
# tools/check_references.allow lists the hits that may stay, one per line: "<path>:<fixed text>". A hit stays when
# its path equals <path> and its line contains <fixed text>. Lines starting with "#" are comments.
# Submodules (third_party/) are not searched, git grep does not enter them.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pattern='armarx|armar[67]|axii|h2t|kit\.edu|/cvhci|/lsdf|/var/tmp|armar-user|visionx|armem'
allow=tools/check_references.allow
[[ -f $allow ]] || { echo "check_references: $allow is missing" >&2; exit 2; }

paths=()
texts=()
while IFS= read -r entry; do
    [[ -z $entry || $entry == \#* ]] && continue
    [[ $entry == *:* ]] || { echo "check_references: bad allowlist line: $entry" >&2; exit 2; }
    paths+=("${entry%%:*}")
    texts+=("${entry#*:}")
done < "$allow"

found=0
while IFS= read -r hit; do
    [[ -z $hit ]] && continue
    path=${hit%%:*}
    rest=${hit#*:}
    line=${rest#*:}
    allowed=0
    for i in "${!paths[@]}"; do
        if [[ $path == "${paths[$i]}" && $line == *"${texts[$i]}"* ]]; then
            allowed=1
            break
        fi
    done
    if (( ! allowed )); then
        echo "$hit"
        found=1
    fi
done < <(git grep -nIiE "$pattern" -- . ':!tools/check_references.sh' ':!tools/check_references.allow' || true)

if (( found )); then
    echo "check_references: the lines above name internal systems or paths" >&2
    exit 1
fi
echo "check_references: no internal references"
