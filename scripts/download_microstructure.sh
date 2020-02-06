# Data from Scripps Microstructure Dataset: https://microstructure.ucsd.edu/#/cruise/33SW19970313
curl "https://cchdo.ucsd.edu/data/13505/bbtre97_microstructure.zip" -C - --create-dirs -o ../data/bbtre97_microstructure.zip

cd ../data
unzip bbtre97_microstructure.zip
rm bbtre97_microstructure.zip