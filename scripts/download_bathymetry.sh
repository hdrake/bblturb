# Download topographic product from: https://topex.ucsd.edu/marine_topo/
#curl "ftp://topex.ucsd.edu/pub/global_topo_1min/topo_19.1.nc" -C - --create-dirs -o ../data/SS97_topography.nc

# Download 15 arcsecond bathymetry from https://topex.ucsd.edu/WWW_html/srtm15_plus.html
curl "ftp://topex.ucsd.edu/pub/srtm15_plus/SRTM15+V2.1.nc" -C - --create-dirs -o ../data/Tozer2019_bathymetry.nc
