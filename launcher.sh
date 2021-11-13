rm -rf venv
python3.7 -m venv ./venv --without-pip
source ./venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
deactivate
source ./venv/bin/activate
pip install numpy

python3.7 a4.py
