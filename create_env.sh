conda create -n handbook python=3.11

conda init
#关掉当前终端
conda deactivate
conda activate handbook

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

cd ~/project
git clone https://github.com/lindafei01/sudoKey.git
cd sudoKey/alignment-handbook

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

pip install -e .
pip install "flash-attn==2.7.4.post1" --no-build-isolation



