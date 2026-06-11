#!/bin/bash

sudo apt update

sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y pulseaudio alsa-utils libasound2-plugins
sudo apt install -y python3-pip

pip3 install -r requirements.txt