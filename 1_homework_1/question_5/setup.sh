# download dataset
if test -d ../../data/cifar-100-python
then
    echo "already downloaded"
else
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    mv cifar-100-python.tar.gz ../../data/
    cd ../../data
    tar -xvf cifar-100-python.tar.gz
    rm cifar-100-python.tar.gz
fi

# download pre-trained model
# if test -f ./mobilenet_v2.pth
# then
#     echo "pre-train model ready"
# else
#     wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
#     mv mobilenet_v2-b0353104.pth mobilenet_v2.pth
# fi