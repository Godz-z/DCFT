# Adapating DeBERTaV3 with DCFT (Coling 2025)

The folder contains the implementation of DCFT in DeBERTaV3 using the updated package of `loralib`, which contains the implementation of DCFT.
Our code is baesd on AdaLoRA-[Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.10512.pdf) (ICLR 2023). 

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

### Citation
```BibTeX
@inproceedings{zhang-etal-2025-parameter,
    title = "Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace",
    author = "Zhang, Jia-Chen  and
      Xiong, Yu-Jie  and
      Xia, Chun-Ming  and
      Zhu, Dong-Hai  and
      Qiu, Xi-He",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    year = "2025",
    address = "Abu Dhabi, UAE",
    url = "https://aclanthology.org/2025.coling-main.265/",
    pages = "3924--3935",
}
```
