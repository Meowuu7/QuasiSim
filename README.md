# NeuS


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
