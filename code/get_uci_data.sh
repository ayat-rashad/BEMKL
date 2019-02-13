export data_path="../data/UCI"
export get_comm="wget -r -nH --cut-dirs=2 --no-parent --reject='index.html*' -P $data_path" 


$get_comm https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/
$get_comm https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
$get_comm https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
