# QuasiSim



overview and make clear what are included in the repo and the future plan


## Getting Started


### Environment setup
TODO: environment setup

(important Diffhand setup) 

Install `torch_cluster`:
```shell
mkdir whls
cd whls
wget https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_cluster-1.6.3%2Bpt22cu121-cp311-cp311-linux_x86_64.whl
pip install torch_cluster-1.6.3%2Bpt22cu121-cp311-cp311-linux_x86_64.whl
```

### Get data 

**Examples data**

TODO: examples data downloading (those you put in the data folder)


**Checkpoints**

<!-- TODO: checkpoints downloading -->
Download [pre-optimized checkpoints](https://1drv.ms/f/s!AgSPtac7QUbHgUNU7vF8a7V0WS9t?e=lp5GSU) and organize them the same way as we do in OnDrive. The expected file structure is as follows: 
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







comparisons between optimizing states and optimizing actions (both just for the kinematic hand tracking)




We present a novel neural surface reconstruction method, called NeuS (pronunciation: /nuːz/, same as "news"), for reconstructing objects and scenes with high fidelity from 2D image inputs.

![](./static/intro_1_compressed.gif)
![](./static/intro_2_compressed.gif)

## [Project page](https://lingjie0206.github.io/papers/NeuS/) |  [Paper](https://arxiv.org/abs/2106.10689) | [Data](https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?dl=0)
This is the official repo for the implementation of **NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction**.

## Usage

#### Data Convention
The data is organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Setup

Clone this repository

```shell
git clone https://github.com/Totoro97/NeuS.git
cd NeuS
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>

  - torch==1.8.0
  - opencv_python==4.5.2.52
  - trimesh==3.9.8 
  - numpy==1.19.2
  - pyhocon==0.3.57
  - icecream==2.1.0
  - tqdm==4.50.2
  - scipy==1.7.0
  - PyMCubes==0.1.2

</details>

### Running

- **Training without mask**

```shell
# CUDA_VISIBLE_DEVICES=0 python exp_runner.py --mode train --conf ./confs/womask.conf --case hand_test
# /data2/datasets/diffsim/neus/public_data/hand_test_routine_2_light_color
python exp_runner.py --mode train --conf ./confs/womask.conf --case <case_name> 

export cuda_ids="6"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color
```

```shell
export cuda_ids="6"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color_wtime


export cuda_ids="5"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color_wtime


export cuda_ids="4"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color_wtime


export cuda_ids="5"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color_wtime

# for bending network #
export cuda_ids="0"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color_wtime




export cuda_ids="7"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask.conf --case hand_test_routine_2_light_color_wtime




export cuda_ids="6"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask_opt_states.conf --case hand_test_routine_2_light_color_wtime

# /home/xueyi/diffsim/NeuS/confs/wmask_extract_corr.conf
export cuda_ids="7"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti.py --mode train --conf ./confs/wmask_extract_corr.conf --case hand_test_routine_2_light_color_wtime


export cuda_ids="1"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_rigidtrans.py --mode train --conf ./confs/wmask_refine_passive_rigidtrans.conf --case hand_test_routine_2_light_color_wtime_active_passive
 # parameterize



# for bending network #
export cuda_ids="2"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs.py --mode train --conf ./confs/wmask_refine_passive.conf --case hand_test_routine_2_light_color_wtime_active_passive
# denoised from the noise space #
# from aunified noise space to #


### with selector ###
export cuda_ids="7"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_motion.py --mode train --conf ./confs/wmask_refine_passive_with_selector.conf --case hand_test_routine_2_light_color_wtime_active_passive


### with selector; with optimized sdf and bending weights ###
export cuda_ids="6"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_motion.py --mode train --conf ./confs/wmask_refine_passive_with_selector.conf --case hand_test_routine_2_light_color_wtime_active_passive


### with selector; with optimized sdf and bending weights ###
export cuda_ids="4"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_motion.py --mode train --conf ./confs/wmask_refine_passive_with_selector.conf --case hand_test_routine_2_light_color_wtime_active_passive


### with selector; with optimized sdf and bending weights; use pre-opt rigid trans ###
export cuda_ids="3"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_motion.py --mode train --conf ./confs/wmask_refine_passive_with_selector.conf --case hand_test_routine_2_light_color_wtime_active_passive
# motion filed and the ###


### with selector; with optimized sdf and bending weights; use pre-opt rigid trans ###
export cuda_ids="7"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_motion.py --mode train --conf ./confs/wmask_refine_passive_with_selector.conf --case hand_test_routine_2_light_color_wtime_active_passive


# for bending network #
export cuda_ids="7"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs.py --mode train --conf ./confs/wmask_refine_passive.conf --case hand_test_routine_2_light_color_wtime_active_passive


# for bending network; for the bending network #
export cuda_ids="5"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs.py --mode train --conf ./confs/wmask_refine_passive_eval.conf --case hand_test_routine_2_light_color_wtime_active_passive


export cuda_ids="5"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs_rigidtrans.py --mode train --conf ./confs/wmask_refine_passive_rigidtrans_eval.conf --case hand_test_routine_2_light_color_wtime_active_passive


# /home/xueyi/diffsim/NeuS/models/renderer_def_multi_objs_rigidtrans_forward.py
# /home/xueyi/diffsim/NeuS/confs/wmask_refine_passive_rigidtrans_forward.conf
# /home/xueyi/diffsim/NeuS/exp_runner_arti_forward.py
export cuda_ids="5"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_forward.py --mode train --conf ./confs/wmask_refine_passive_rigidtrans_forward.conf --case hand_test_routine_2_light_color_wtime_active_passive


export cuda_ids="1"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_forward.py --mode train --conf ./confs/wmask_refine_passive_rigidtrans_forward.conf --case hand_test_routine_2_light_color_wtime_active_passive


export cuda_ids="0"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${cuda_ids} python exp_runner_arti_multi_objs.py --mode train --conf ./confs/wmask_refine_passive.conf --case hand_test_routine_2_light_color_wtime_active_passive
```

- **Training with mask**

```shell
python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name>
```

- **Extract surface from trained model** 

```shell
python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

- **View interpolation**

```shell
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding image set of view interpolation can be found in `exp/<case_name>/<exp_name>/render/`.

### Train NeuS with your custom data

More information can be found in [preprocess_custom_data](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data).

## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{wang2021neus,
  title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
  author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2106.10689},
  year={2021}
}
```


## Acknowledgement

Some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). Thanks for these great projects.
