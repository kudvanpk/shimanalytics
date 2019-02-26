#!/bin/bash

# bash /opt/tsatools/Anaconda-2.1.0-Linux-x86_64.sh -b 
# source ~/.bashrc
# export PATH=/root/anaconda/bin:$PATH
which python
python toplevel.py --outfile "test" --alchemy_env "prod" --monitors monitor.json --creds creds.json 
