# Download topographic product from: https://topex.ucsd.edu/marine_topo/
curl "ftp://topex.ucsd.edu/pub/global_topo_1min/topo_19.1.nc" -C - --create-dirs -o ../data/SS97_topography.nc
