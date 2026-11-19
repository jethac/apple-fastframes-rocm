#!/usr/bin/env bash
# run.sh SCRIPT ARGS... : run under the private Ryzen AI 1.9 runtime with scoped libs
cd "$(dirname "$0")"
exec bash repro/run-sdk.sh runtime/opt/amd/ryzenai/1.9.0 venv/bin/python "$@"
