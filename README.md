# Adapating DeBERTaV3 with DCFT

The folder contains the implementation of DCFT in DeBERTaV3 using the updated package of `loralib`, which contains the implementation of DCFT.


## Setup Environment

### Create and activate the conda env
```bash
conda create -n NLU python=3.7
conda activate NLU 
```

### Install Pytorch
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install the pre-requisites
Install dependencies: 
```bash
pip install -r requirements.txt
```

Install `transformers`: (here we fork NLU examples from [microsoft/LoRA](https://github.com/microsoft/LoRA/tree/main/examples/NLU) and build our examples based on their `transformers` version, which is `v4.4.2`.)
```bash
pip install -e . 
```

Install the updated `loralib`:
```bash
pip install -e ../loralib/
```

