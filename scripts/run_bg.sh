#!/usr/bin/env bash
# Run a command detached from the shell (survives the tool-call time limit). Log -> results/logs/<name>.log
name=$1; shift
cd "$(dirname "$0")/.."
mkdir -p results/logs
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages
setsid nohup "$@" > "results/logs/$name.log" 2>&1 < /dev/null &
echo "started $name (pid $!)"
