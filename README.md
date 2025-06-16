
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
- For multiple model SR images used in `img_combine.py`, refer to the provided download link.
