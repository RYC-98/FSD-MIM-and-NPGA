# Details about FSD-MIM and FD-FSD-MIM will be completed as soon as the paper is accepted



## Requirements

- python 3.8
- torch 1.8
- numpy 1.19
- pandas 1.2


## Implementation

- **Generate adversarial examples**

  Using `npga.py` to generate Imperceptible adversarial examples,  you can run this attack as following
  
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python npga.py --output_dir outputs
  ```
  
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
![Results1](https://github.com/RYC-98/FSD-MIM-and-NPGA/blob/main/result1.png)

![Results2](https://github.com/RYC-98/FSD-MIM-and-NPGA/blob/main/result2.png)

