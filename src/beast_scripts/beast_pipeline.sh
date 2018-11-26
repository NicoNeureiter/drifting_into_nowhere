#!/usr/bin/sh

HPD=$1
BURNIN=$2
CWD=$3
GEOJSON=$4

cd $CWD
#export BEAST="/opt/phylo/beast"
#export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
#export PKG_CONFIG_PATH=$HOME/lib/pkgconfig:$PKG_CONFIG_PATH

# Generate posterior samples
beast -overwrite nowhere.xml

# Combine posterior samples
treeannotator -burnin $BURNIN -hpd2D 0.$HPD nowhere.trees nowhere.tree

## Generate json of spatial development
#java -jar "/opt/phylo/spreaD3_v0.9.6.jar" -parse -tree nowhere.tree \
#    -xCoordinate location2 -yCoordinate location1 \
#    -HPD $HPD -externalAnnotations true -mrsd 2000.00 \
#    -geojson "$CWD/$GEOJSON" \
#    -output nowhere.json
#
## Render json for 2D browser visualization
#cd /opt/phylo
#java -jar spreaD3_v0.9.6.jar -render d3 -json "$CWD/nowhere.json" -output "$CWD/nowhere"