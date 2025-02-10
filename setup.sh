pip install -U torch torchvision torchaudio

pip install -U pyqt5-sip pyqtchart pyqtwebengine roma timm pycolmap

if [ ! -d "/usr/include/opencv2/" ]; then
  sudo ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2
fi

cd ./ace_zero/Lib/dsacstar/
python setup.py install
