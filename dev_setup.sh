cd ..
git clone https://github.com/nianticlabs/acezero.git ace0

conda install -c conda-forge opencv

pip install opencv-python tqdm joblib scipy scikit-image \
  pyrender matplotlib

pip install \
  pyqt5-sip==4.19.18 \
  pyqtchart==5.12 \
  pyqtwebengine==5.12.1 \
  roma==1.4.1 \
  timm==0.6.7 \
  pycolmap==0.4.0

cd ../ace0/dsacstar
python setup.py install
