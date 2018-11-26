#!/usr/bin/env bash

HPD=$1
BURNIN=$2
CWD=$3

cd $CWD
export BEAST="/opt/phylo/beast"
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/lib/pkgconfig:$PKG_CONFIG_PATH

treeannotator -burnin $BURNIN -hpd2D 0.$HPD nowhere.trees nowhere.tree