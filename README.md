# Hierarchical-Attentioin-Network

### paper: [16HLT-hierarchical-attention-networks.pdf](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)

### Introduction
- Đây là code triển khai kiến trúc Hierarchical-Attention-Network dựa trên [jaehunjung1](https://github.com/jaehunjung1/Hierarchical-Attention-Network)
- Nhiệm vụ là phân loại văn bản.
- Dataset sử dụng: [The_20_newsgroups_text](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
- Model đã được training trên Dataset (với chỉ 5 classes) trong vòng 20 epochs và đạt kết quả:
  ![image](https://github.com/tandat17z/Hierarchical-attention-networks/assets/126872123/3edd79a6-4486-4abb-9db3-d271d76c81ab)
- Quá trình training: [Colab](https://colab.research.google.com/drive/1ivRVcTm_Lfal6974JEocGIMt57j1JCtk?usp=sharing)

### usage
- Download and add `glove.6B.100d.txt` into `./data/glove/`
- Run `python train.py <args>

### References:
- [jaehunjung1/Hierarchical-Attention-Network](https://github.com/jaehunjung1/Hierarchical-Attention-Network)
