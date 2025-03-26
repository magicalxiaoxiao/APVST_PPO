
# Rate Splitting Multiple Access-Enabled Adaptive Panoramic Video Semantic Transmission

PyTorch Implementation of the Paper "Rate Splitting Multiple Access-Enabled Adaptive Panoramic Video Semantic Transmission"

arXiv Link: [https://arxiv.org/abs/2402.16581](https://arxiv.org/abs/2402.16581)

---

## **Prerequisites**
* Python 3.8 and [Conda](https://www.anaconda.com/)
* CUDA 11.7
* Environment
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install compressai==1.2.0
pip install timm==0.9.7
pip install imageio==2.31.5
```
---

## **APVST Semantic Model**  
   A semantic model named **APVST** designed for the semantic transmission of panoramic videos.

### 1.Download APVST Pretrained Model

Download [APVST_best_model](https://drive.google.com/drive/folders/12gAd8988ylMfkrvNw1uLUcRdhX-xVxNI?usp=sharing) (optimized for WMSE) and put it into `APVST/ckpt` folder.

### 2.Usage 

Example to test the APVST semantic model:

```bash
cd APVST_PPO/APVST
python3 test_demo.py
```
where `APVST_PPO/APVST/test_image/00001.png` and `APVST_PPO/APVST/test_image/00000.png` denote the current panoramic frame and the reference panoramic frame, respectively.

---

## **DRL_PPO Resource Allocation**  
A resource allocation system based on the PPO (Proximal Policy Optimization) algorithm using deep reinforcement learning.

---

## **TODO List**

- [x] Release APVST Network Structure
- [x] Release APVST Pretrained Model
- [x] Release DRL_PPO Environment & Network Structure
- [ ] Release APVST Training Code
- [ ] Release DRL_PPO Training Code
- [ ] Release All Pretrained Models

---
## **Contact Us**
Haixiao Gao: [haixiao@bupt.edu.cn](mailto:haixiao@bupt.edu.cn)

---
## **Acknowledgements**
Codebase built upon [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer) and [CompressAI](https://github.com/InterDigitalInc/CompressAI/), and [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22/).

---
## **Citation**
If you find the code helpful in your research or work, please cite:
```
@article{gao2024rate,
  title={Rate Splitting Multiple Access-Enabled Adaptive Panoramic Video Semantic Transmission},
  author={Gao, Haixiao and Sun, Mengying and Xu, Xiaodong and Han, Shujun and Wang, Bizhu and Zhang, Jingxuan and Zhang, Ping},
  journal={arXiv preprint arXiv:2402.16581},
  year={2024}
}
```
