#!/usr/bin/env bash

HPD=$1
BURNIN=$2
CWD=$3
TREES_FILE=$4
MCC_FILE=$5

cd $CWD
#export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
#export PKG_CONFIG_PATH=$HOME/lib/pkgconfig:$PKG_CONFIG_PATH
export BEAST1="/opt/phylo/beast_1.10"

treeannotator -burnin $BURNIN -hpd2D 0.$HPD $TREES_FILE $MCC_FILE