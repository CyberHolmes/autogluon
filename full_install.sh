sudo apt-get update
sudo apt-get install gcc python3-dev
pip install torch
pip install torchvision
pip install torchtext
pip install mxnet
pip install -e core/
pip install -e tabular/
pip install -e mxnet/
pip install -e extra/
pip install -e text/
pip install -e vision/
pip install -e autogluon/
mkdir -p benchmarking/output/jct
