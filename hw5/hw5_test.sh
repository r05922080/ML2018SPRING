#/usr/bin bash
wget 'https://www.dropbox.com/s/ezzr44ph4kfr8ot/model.h5?dl=1' -O "model.h5"
python3 hw5_test.py $1 $2 $3
