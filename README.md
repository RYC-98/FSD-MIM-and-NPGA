# Frequency-based methods for improving the imperceptibility and transferability of adversarial examples



## Requirements

- python 3.8
- torch 1.8
- numpy 1.19
- pandas 1.2


## Implementation

- **Generate adversarial examples**

  
  Using `FSD_MIM.py` to generate highly transferable adversarial examples,  you can run this attack as following
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python FSD_MIM.py --output_dir outputs
  ```
  where `gpuid` can be set to any free GPU ID in your machine. And adversarial examples will be generated in directory `./outputs`.
  
- **Evaluations on normally trained models**

  Running `verify.py` to evaluate the attack  success rate

  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python verify.py
  ```
  
## Main Results
Number of augmented copies is set to 5 for each iteration.
![Results1](https://github.com/RYC-98/FSD-MIM-and-NPGA/blob/main/f1.png)

![Results2](https://github.com/RYC-98/FSD-MIM-and-NPGA/blob/main/f2.png)

