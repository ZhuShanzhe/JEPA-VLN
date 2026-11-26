#!/usr/bin/env bash
adapter="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
box_release="$(dirname "$adapter")"
box_project="$(dirname "$box_release")"
source "$box_project/jepa_reid_v25/integration/environment.sh"
export PYTHONPATH="$adapter:$box_release/refine:$box_release/compare:$box_release/sam:$PYTHONPATH"
