data_path="../data"


wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz -P $data_path"/17_cat_flower"
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz -P $data_path
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/distancematrices17itfeat08.mat -P $data_path
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat -P $data_path

cd $data_path
tar -zxvf `*.tgz`

# Dataset2
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -P $data_path"/102_cat_flower"
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -P $data_path"/102_cat_flower"
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/distancematrices102.mat -P $data_path"/102_cat_flower"
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/setid.mat -P $data_path"/102_cat_flower"

cd $data_path
tar -zxvf `*.tgz`
