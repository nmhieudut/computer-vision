rm -r ./output/negative
rm -r ./output/positive

mkdir ./output/negative
mkdir ./output/positive

# hieu
python H:/ComputerVision/Smile_Detection/detect_smile.py -c H:/ComputerVision/Smile_Detection/haarcascade_frontalface_default.xml -m H:/ComputerVision/Smile_Detection/output/lenet.hdf5