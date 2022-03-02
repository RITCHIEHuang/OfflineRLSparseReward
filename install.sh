#!/bin/bash
# install d4rl
cd d4rl/ && pip install -i https://mirrors.aliyun.com/pypi/simple -e .
# install neorl
cd ../neorl/ && pip install -i https://mirrors.aliyun.com/pypi/simple -e .
# install OfflineRL
cd ../OfflineRL/ && pip install -i https://mirrors.aliyun.com/pypi/simple -e .
# install recsim 
cd ../recsim/ && pip install -i https://mirrors.aliyun.com/pypi/simple -e .
# install project
cd ../ && pip install -i https://mirrors.aliyun.com/pypi/simple -e .