
## Requirements

- python 3.8
- torch 1.8
- numpy 1.19
- pandas 1.2


## Implementation

- **Prepare models**

  Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into `./models/`

- **Generate adversarial examples**

  Using `FSD_MIM.py` to generate adversarial examples,  you can run this attack as following
  
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python FSD_MIM.py --output_dir outputs
  ```
  where `gpuid` can be set to any free GPU ID in your machine. And adversarial examples will be generated in directory `./outputs`.
  
- **Evaluations on normally trained models**

  Running `verify.py` to evaluate the attack  success rate

  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python verify.py
  ```



