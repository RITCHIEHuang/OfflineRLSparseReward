#!/bin/bash
# # install d4rl
# cd d4rl/ && pip install -e .
# # install neorl
# cd ../neorl/ && pip install -e .
# # install OfflineRL
# cd ../OfflineRL/ && pip install -e .
# # install recsim 
# cd ../recsim/ && pip install -e .
# # install project
# cd ../ && pip install -e .

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