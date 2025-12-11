Using WSL2 Ubuntu-24

Chạy các lệnh sau đây trên terminal

sudo apt update
sudo apt upgrade -y

# C++ compiler, make, gdb, git, cmake
sudo apt install -y build-essential git cmake gdb

# Python để sau này vẽ biểu đồ, xử lý log nếu thích
sudo apt install -y python3 python3-pip
pip3 install --user numpy matplotlib

sudo apt install -y nvidia-cuda-toolkit
