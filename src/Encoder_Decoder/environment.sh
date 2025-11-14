wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh
source .bashrc


conda create -n p5 python=3.9.11
conda activate p5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers==4.26.0 wandb tqdm datasets scikit-learn==1.1.2 nvitop

apt update
apt install tmux