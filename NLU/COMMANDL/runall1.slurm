#!/bin/bash

### 设置该作业的作业名
#SBATCH --job-name=test-nvidia

### 指定申请的节点
#SBATCH --nodelist=gpu01

### 指定该作业需要1个节点数
#SBATCH --nodes=1

### 该作业需要1个CPU
#SBATCH --ntasks=1

### 申请1块GPU卡
#SBATCH --gres=gpu:1


source ~/.bashrc
source activate icnovlora
bash ./COMMANDL/runall.sh



