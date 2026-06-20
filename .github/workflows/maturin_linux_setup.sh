#!/usr/bin/env bash
# Cranelift backend — no LLVM toolchain required. Only SuiteSparse is installed
# (optional `suitesparse` linear-solver feature).
if [ -f "/etc/debian_version" ]; then
    echo Installing SuiteSparse via apt
    apt-get update
    apt-get install -y libsuitesparse-dev
else
    echo Installing SuiteSparse via yum
    yum update -y
    yum install -y suitesparse-devel
fi
