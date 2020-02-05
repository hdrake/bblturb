# Download topographic product from https://www.gebco.net/data_and_products/gridded_bathymetry_data/
curl "https://www.bodc.ac.uk/data/open_download/gebco/GEBCO_15SEC/zip/" -C - --create-dirs -o ../data/bathymetry.zip
cd ../data/
unzip bathymetry.zip
rm bathymetry.zip
