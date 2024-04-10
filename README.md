# QuasiSim



### [Project](https://meowuu7.github.io/GeneOH-Diffusion/) | [Gradio Demo](https://huggingface.co/spaces/xymeow7/quasi-physical-sims) | [Video]()

The implementation of the paper [**QuasiSim**](https://meowuu7.git**hub.io/GeneOH-Diffusion/), presenting a parameterized quasi-physical simulator for transferring kinematics-only human manipulation demonstrations to a simulated dexterous robot hand. 
<!-- presenting a ***generalizable HOI denoising model*** designed to ***curate high-quality interaction data***. -->



https://github.com/Meowuu7/QuasiSim/assets/50799886/44233442-3382-4de8-8dbc-9e48b2b6c271

The repository contains 
- Analytical part of the parameterized quasi-physical simulator; 
- Detailed instructions on the optimization process for a manipulation sequence example (first two stages). 

We will add the remaining code and instructions on the last optimization stage, as well as the data and more manipulation examples. 
<!-- We will add the data and the evaluation process for the remaining test datasets, as well as the training procedure. These updates are expected to be completed before May 2024. -->


## Getting Started


This code was tested on `Ubuntu 20.04.5 LTS` and requires:

* Python 3.8.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)


### Environment setup

<!-- TODO: environment setup -->




<!-- (important Diffhand setup)  -->

1. Creat a virtual environment and install necessary dependencies

Create a virtual environment

```shell
conda create -n quasisim python==3.8.8
conda activate quasisim
```

2. Install `torch2.2.0+cu121`
```shell
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```


3. Install `torch_cluster`:
```shell
mkdir whls
cd whls
wget https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_cluster-1.6.3%2Bpt22cu121-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.6.3+pt22cu121-cp38-cp38-linux_x86_64.whl
```

4. Install other dependences

```shell
pip install -r requirements.txt
```


5. Setup [DiffHand](https://github.com/eanswer/DiffHand):
```shell
cd DiffHand
cd core
python setup.py install
### For testing the installation ###
cd ..
cd examples
python test_redmax.py
```
It's better to install it from this project since we have made modifications to support our purpose.


### Get data 

**Examples data**

<!-- TODO: examples data downloading (those you put in the data folder) -->
Download the [example data](https://1drv.ms/f/s!AgSPtac7QUbHgVE5vMBOAUPzxxsV?e=B5V6mo) and organize them in the same way as we do in OnDrive. The expected file structure is as follows: 
```shell
data
 |-- grab
     |-- 102
         |-- 102_obj.npy
         |-- 102_obj.obj
         |-- 102_sv_dict.npy
         |-- 102_sv_dict_st_0_ed_108.npy
     ...
```



**Checkpoints**

<!-- TODO: checkpoints downloading -->
Download [pre-optimized checkpoints](https://1drv.ms/f/s!AgSPtac7QUbHgUNU7vF8a7V0WS9t?e=lp5GSU) and organize them in the same way as we do in OnDrive. The expected file structure is as follows: 
```shell
ckpts
 |-- grab
     |-- 102
         |-- xxx1.pt
         |-- xxx2.pt
     ...
```


**URDFs and others**

<!-- TODO: miscs data downloading -->


Download [rsc.zip](https://1drv.ms/u/s!AgSPtac7QUbHgUKUL6O4E7_0ygNT?e=PZlb0I) and [raw_data.zip](https://1drv.ms/u/s!AgSPtac7QUbHgUEIHkPdmjmHUMQc?e=Q9IYY1). Extract them in the root folder of the project. 




## Example Usage

<!-- In the current stage, this repo mainly contains source code on the analytical p -->
We include data, detailed instructions, and result of an example, aiming at present the inverse dynamics problem in the contact-rich manipulation scenario that we can leverage QuasiSim to solve and the optimization process. Currently we release the analytical part of QuasiSim and the first and second optimization stages. The thrid stage along with more examples will be added. 

For the example sequence `data/grab/102` showing a human hand rotating a mouse, the human manipulaton demonstration, tranferred manipulation to the simulated Shadow hand in the QuasiSim's analytical environment, and the manipulation optimized in the Bullet simulator are shown as follows. 


|        Human Manipulation        |       Transferred to Shadow         |         Transferred to Shadow in Bullet         |
| :----------------------: | :---------------------: | :-----------------------: |
| ![](assets/human-1.gif) | ![](assets/robo-1.gif) | ![](assets/robo-bullet-1.gif) |

The following instructions aims at optimizing for the control trajectory that can drive  the Shadow hand to complete the tracking task (as shown in the middle demo) in the stiffest analytical environment of QuasiSim. 

### Stage 1
> **Transferring human demonstrations via point set.** In this stage, we represent the dynamics MANO model and the Shadow model in simulation as parameterized point set. The contact model is tightened to the softest level. The goal is optimizing for a point set trajectory of the Shadow hand to complete the manipulation tracking. 

This stage is divided into three steps as described follows. 

**Step 1: Optimizing for a dynamic simulated MANO hand trajectory**

In this step, we optimize for a control trajectory for the dynamic MANO hand model to track the reference manipulation. Run the following commands sequentially:

```shell
bash scripts_new/train_grab_mano.sh
bash scripts_new/train_grab_mano_wreact.sh
bash scripts_new/train_grab_mano_wreact_optacts.sh
```

**Step 2: Optimizing for a control trajectory for the point set constructed from the MANO hand** 

Run the following four commands sequentially for this step: 

```shell
bash scripts_new/train_grab_pointset_points_dyn_s1.sh
bash scripts_new/train_grab_pointset_points_dyn_s2.sh
bash scripts_new/train_grab_pointset_points_dyn_s3.sh
bash scripts_new/train_grab_pointset_points_dyn_s4.sh
```

**Step 3: Optimizing for a kinematic Shadow hand trajectory**

In this step, we optimize for a kinematic Shadow hand trajectory based on the keypoint based correspondences and mesh surface point based correspondences. Execute the following command for this step: 

```shell
bash scripts_new/train_grab_sparse_retar.sh
```

**Step 4: Optimizing for a control trajectory for the point set constructed from the simulated Shadow hand** 

Execute the following commands for this step:

```shell
bash scripts_new/train_grab_pointset_points_dyn_retar.sh
bash scripts_new/train_grab_pointset_points_dyn_retar_pts.sh
```





### Stage 2
> **Tracking via a contact curriculum.** In this stage, we optimizing for a control trajectory of the simulated Shadow hand to complete the tracking task through a series of contact models. Initially, the contact model is tuned to the sofest level. We then gradually adjust parameters for tuning it to the stiffest level. 

Run the following command: 
```shell
bash scripts_new/train_grab_stage_2_dm_curriculum.sh
```




## TODOs

- [x] Analytical part of QuasiSim
- [x] Optimization example
- [ ] More examples
- [ ] Full version of QuasiSim
- [ ] Custimizing your optimization



## Contact

Please contact xymeow7@gmail.com or create a github issue if you have any questions.


## Bibtex
If you find this code useful in your research, please cite:

```bibtex
@inproceedings{liu2024geneoh,
   title={GeneOH Diffusion: Towards Generalizable Hand-Object Interaction Denoising via Denoising Diffusion},
   author={Liu, Xueyi and Yi, Li},
   booktitle={The Twelfth International Conference on Learning Representations},
   year={2024}
}
```


## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on: [DiffHand](https://github.com/eanswer/DiffHand) and [NeuS](https://github.com/Totoro97/NeuS).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

