# Docker
## Install
```bash
sudo apt update
sudo apt -y upgrade
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce
sudo systemctl status docker
sudo systemctl enable docker
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG
sudo reboot
```
## Usage
- Build
```bash
docker build --rm --tag yuriapxlt/tfc:dev-v0.0 --file Dockerfile empty
```
- Remove
```bash
docker rmi -f yuriapxlt/tfc:dev-v0.0 
```
- Run
```
docker run -ti yuriapxlt/tfc:dev-v0.0 
```
