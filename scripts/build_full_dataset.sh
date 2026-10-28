#!/bin/bash
# Full synthetic dataset for Restage v1: disjoint seed ranges per split, all genres, 8-20 s clips.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m restage.data.make_dataset --n 120 --split test --render-workers 5
$PY -m restage.data.make_dataset --n 120 --split val --render-workers 5
$PY -m restage.data.make_dataset --n 2400 --split train --render-workers 5
du -sh data/train data/val data/test
