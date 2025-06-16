
# **Light Field Super-Resolution**

## **Requirements**
To set up the environment, install the following dependencies:

- **PyTorch**: `2.1.1`
- **Torchvision**: `0.16.1`
- **Python**: `3.9.19`
- **CUDA**: `11.8`
- **Additional Packages**:
  - `causal-conv1d==1.1.1`
  - `mamba-ssm==1.0.1`

### **Installation**
You can create the environment using:
```bash
pip install torch==2.1.1 torchvision==0.16.1
pip install causal-conv1d==1.1.1 mamba-ssm==1.0.1
```

---

## **Training**
### **1. Prepare Training Data**
We utilize five Light Field  benchmarks from [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR):
- **EPFL**
- **HCInew**
- **HCIold**
- **INRIA**
- **STFgantry**

Download the datasets and place them in the `./datasets/` directory.

To generate the training data, run:
```bash
python Generate_Data_for_Training_aug.py
```
The processed training data will be saved in `./data_for_training/`.

### **2. Start Training**
To train the network, run:
```bash
python train.py
```
Model checkpoints will be saved in `./log/`.

---

## **Testing**
### **1. Prepare Test Data**
Generate the test data by running:
```bash
python Generate_Data_for_Test.py
```
The processed test data will be saved in `./data_for_test/`.

### **2. Start Testing**
Perform testing on each dataset using:
```bash
python test.py
```
The output `.mat` files will be stored in `./Results/`.

To generate SR RGB images, run:
```bash
python GenerateResultImages.py
```
The generated images will be saved in `./SRimage/`.

To create combined images, run:
```bash
python img_combine.py
```
The results will be saved in:
- **Real test data:** `./combined_Test_Real/`
- **Synthetic test data:** `./combined_Test_Synth/`

---

## **Notice**
- Please modify the file loading and saving paths as needed to match your specific setup.  
- The pretrained model weights can be downloaded from the link below:

🔗 [Click here to download](https://drive.google.com/file/d/1dkasatZmbp2ltiFoRT04OraAmri0g8eF/view?usp=drive_link).

## 📖 Citation

If you find this work helpful, please consider citing the following paper:

```bibtex
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Haosong and Zhu, Xiancheng and Zeng, Huanqiang and Zhu, Jianqing and Shi, Yifan and Chen, Jing and Hou, Junhui},
    title     = {LFTramba: Comprehensive Information Learning for Light Field Image Super-Resolution via A Hybrid Transformer-Mamba Framework},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {1137-1147}
}
```

## 🙏 Acknowledgements

This code borrows heavily from [BigEPIT](https://github.com/chaowentao/BigEPIT) and [LFMamba](https://github.com/stanley-313/LFMamba) repository. Thanks a lot.

If you have any questions, please pull an Issue and feel free to contact me at:

- 📧 hsliu@stu.hqu.edu.cn  

