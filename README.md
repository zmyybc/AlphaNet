<p>
  <img src="./old_README/logo.png" alt="Your Project Logo" width="60" height="auto">
</p>

# AlphaNet

We present **AlphaNet**, a local frame-based equivariant model designed to tackle the challenges of achieving both accurate and efficient simulations for atomistic systems.  **AlphaNet** enhances computational efficiency and accuracy by leveraging the local geometric structures of atomic environments through the construction of equivariant local frames and learnable frame transitions. And inspired by Quantum Mechanics, AlphaNet **introduces efficient multi-body message passing by using contraction of matrix product states** rather than common 2-body message passing.  Notably, AlphaNet offers one of the best trade-offs between computational efficiency and accuracy among existing models. Moreover, AlphaNet exhibits scalability across a broad spectrum of system and dataset sizes, affirming its versatility.

## News(26.7.26)
We have an accelerated version using triton, see in the **lammps** branch and [the pr](https://github.com/zmyybc/AlphaNet/pull/13). 

If you intend to use AlphaNet in LAMMPS, please switch to the **lammps** branch, we are actively working on merging them into the main branch:
```bash
   git clone https://github.com/zmyybc/AlphaNet.git
   git checkout lammps 
   ```

## Update Log (v0.1.2)

### Major Changes

1. **Added new 2 pretrained models**
   - Provide a pretrained model for materials: **AlphaNet-MATPES-r2scan** and our first pretrained model for catlysis: **AlphaNet-AQCAT25**, see them in the [pretrained](./pretrained) folder.
   - Users can **convert the checkpoint trained in torch to our JAX model**
   
2. **Fixed some bugs**
   - Support non-periodic boundary conditions in our ase calculator.
   - Fixed errors in float64
     

## Installation Guide

### Installation Steps

1. **Create a Conda Environment**

   Open your terminal or command prompt and run:

   ```bash
   conda create -n alphanet_env python=3.8 #or later version
   ```

2. **Activate the Environment**

   ```bash
   conda activate alphanet_env
   ```

3. **Install Required Packages**

   Navigate to your desired installation directory and run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Clone the Repository**

   ```bash
   git clone https://github.com/zmyybc/AlphaNet.git
   ```

5. **Install AlphaNet**

   Navigate into the cloned repository and install AlphaNet in editable mode:

   ```bash
   cd AlphaNet
   pip install -e .
   ```

   This allows you to make changes to the codebase and have them reflected without reinstalling the package.


## 💡 Key Tips for AlphaNet Training

---

### 🚨 Training Settings

* **Gradient Clipping:**
    * If you train AlphaNet in your own code, it is **important to turn on gradient clipping**.

* **Weight Decay:**
    * Currently, please set **weight decay to $0$**.

---

### 📉 Loss Weight Adjustment Strategy

The setting for loss weights depends on the **standard deviation (std) of energy per atom** in your dataset.

| Data Type | Std of Energy per Atom | Recommendation | Initial Custom Settings |
| :--- | :--- | :--- | :--- |
| **Common (e.g., VASP)** | Below $1 \times 10^3$ (or slightly larger) | Use **default settings**. | N/A |
| **Large Fluctuations (e.g., Gaussian, CP2K)** | Large | **Manual Adjustment** | **Energy Weight:** $0.1$ |
| | | | **Forces Weight:** $100$ |

#### **Dynamic Adjustment Workflow (For Large Std Data)**

1.  **Initial Phase:** Set the Energy:Forces weight ratio to **$0.1:100$** and use a learning rate of **$1 \times 10^{-4}$**.
2.  **Monitor:** Watch for the loss to start decreasing.
3.  **Adjust:** Once the loss goes down, you should **manually and gradually**:
        * Cut down the **learning rate**.
        * Increase the weight of **energy**.

> ℹ️ **Note:** We are actively working on implementing this dynamic adjustment automatically in future updates.

---

## Quick Start

### Basic Usage

The settings are put into a config file, you can see the json files provided as example, or see comments in `alphanet/config.py` for some help. 

In this version, you can set **"zbl" in the "model" field to true to enable ZBL potential**.

Our code is based on pytorch-lightning, and in this version we provide command line interaction, which makes AlphaNet easier to use. However if you are already familar with python and torch, which is not that hard, it would be great to use the model in a torch way and do further exploration. 


In all there are 3 commands:
1. Train a model:

```bash 
alpha-train example.json # use --help to see more functions, like multi-gpu training resuming from ckpt...
```
2. Convert from lightning ckpt to state_dict ckpt:
```bash 
alpha-conv -i in.ckpt -o out.ckpt # use --help to see more functions
```
3. Evaluate a model and draw diagonal plot:
```bash 
alpha-eval -c example.json -m /path/to/ckpt # use --help to see more functions
```
The functions above can also be used in a script way like previous version, see `old_README`.

### Make dataset
To prepare the training dataset in format of pickle, you can use:

1. from deepmd:

```bash 
python scripts/dp2pic_batch.py
```

2. from extxyz:

```bash 
python scripts/xyz2pic.py
```

So if you work in AlphaNet directory, the dataset should be organized as:
```
AlphaNet/
├── input.json
└── dataset/
    ├── my_dataset_1/ #This is your self-decided name, which should also written in your json file
    │   ├── raw/
    │   └── processed/ # would appear after you first run training, when you need to change the dataset, you should remove it
    ├── my_dataset_2/ #This is your self-decided name, which should also written in your json file
    │   ├── raw/
    │   └── processed/
    └── custom_dataset/#This is your self-decided name, which should also written in your json file
        ├── raw/
        └── processed/
```
There is also an ase calculator, you can use it in jax or torch in this:

```python 
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.infer.new_haiku import AlphaNetCalculator #JAX version
from alphanet.config import All_Config
from ase.build import bulk
# example usage
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)

calculator = AlphaNetCalculator(
        ckpt_path='./alex_0410.ckpt',#./pretrained/OMA/haiku/haiku_params.pkl haiku ckpt
        device = 'cuda',
        precision = '32',
        config=All_Config().from_json('./pretrained/OMA/oma.json'),
)

atoms.calc = calculator
print(atoms.get_potential_energy())

```

### Using AlphaNet in JAX
1. Installation
   ```bash
   pip install -U --pre jax jaxlib "jax-cuda12-plugin[with-cuda]" jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
   ```
   This is just for reference. JAX installation may be tricky, please get more information in [JAX](https://docs.jax.dev/en/latest/installation.html) and its github issues.

   Currently I suggest **version>=0.4 <=0.4.10 or >=0.4.30 <=0.5 or ==0.6.2**  

   Install flax and haiku
   ```bash
   pip install matscipy
   pip install flax
   pip install -U dm-haiku
   ```

2. Converted checkpoints:
   
   See pretrained directory

3. Convert a self-trained ckpt
   
   First from torch to flax:
   ```bash
   python scripts/conv_pt2flax.py #need to modify the path in it.
   ```
   Then from flax to haiku:

   ```bash
   python scripts/flax2haiku.py #need to modify the path in it.
   ```
     
4. Performance:
   
   The output (energy forces stress) difference from torch model would below 0.001. I ran speed tests on a 4090 GPU, system size from 4 to 300, and get a **2.5x to 3x** speed up.

   Please note jax model need to be compiled first, so the first run could take a few seconds or minutes, but would be pretty fast after that.

## Self-made dataset



[The Zeolite Dataset](https://doi.org/10.6084/m9.figshare.27800211)


## Pretrained Models

Current pretrained models:

For materials:
- [AlphaNet-MPtrj-v1](pretrained/MPtrj): A model trained on the MpTrj dataset.
- [AlphaNet-oma-v1](pretrained/OMA): A model trained on the OMAT24 dataset, and finetuned on sALEX+MPtrj.
- [AlphaNet-MATPES-r2scan](pretrained/MATPES): A model trained on the MATPES-r2scan dataset.

For surfaces adsorbtion and reactions:
- [AlphaNet-AQCAT25](pretrained/AQCAT25): A model trained on the AQCAT25 dataset.

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank all contributors and the community for their support. Please open an issue or disscusion  if there are any problems. 

## Citation
```bash
@article{yin2025alphanet,
  title={{AlphaNet}: scaling up local-frame-based neural network interatomic potentials},
  author={Yin, Bangchen and Wang, Jiaao and Du, Weitao and Wang, Pengbo and Ying, Penghua and Jia, Haojun and Zhang, Zisheng and Du, Yuanqi and Gomes, Carla and Duan, Chenru and Henkelman, Graeme and Xiao, Hai},
  journal={npj Computational Materials},
  volume={11},
  number={1},
  pages={332},
  year={2025},
  publisher={Nature Portfolio}
}
```













