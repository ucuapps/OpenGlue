# Configuration details
Config files contain the model hyperparameters. Here we describe every parameter along with its default value.

Our config is organized as follows:

* `gpus` (List[int, ...]) - ids of gpus to run training on, default=`[0, 1]` 
  

* `data`:  
  * `root_path` (str) - root directory for MegaDepth data
  * `train_list_path` (str) - path to the file with train split, ex.: megadepth_train_2.0.txt
  * `val_list_path` (str) - path to the file with validation split, ex.: megadepth_valid_2.0.txt
  * `test_list_path` (str) - path to the file with test split
  * `batch_size_per_gpu` (int) -  number of image pairs in a batch per gpu, default=`4` 
  * `dataloader_workers_per_gpu` (int) - number of workers per one gpu, default=`4` 
  * `target_size` (List[int, int]) - shape of input image after resizing, default=`[ 960, 720 ]`
  * `val_max_pairs_per_scene` (int) - max number of image pairs retrieved from the same scene, default=`50`
    
  * `features_dir` --optional-- [ONLY if features were CACHED] (str) - path to the saved cached features directory
  * `max_keypoints` --optional-- [ONLY if features were CACHED] (int) - max number of keypoints to detect, default=`1024`

* `logging`:
    * `root_path` (str) - directory, where experiment's logs will be saved
    * `name` (str) - experiment name
    * `val_frequency` (int) - number of iterations for frequency of computing validation, default=`10000`
    * `train_logs_steps` (int) - number of iterations for frquency of logging results on train, default=`50`

  
* `train`: 
    * `grad_clip` (float) - clip gradients' global norm to <= threshold, default=`10.0` 
    * `precision` (int) - mixed precision configuration, combines the use of both 32 and 16 bit floating points, default=`32`
    * `gt_positive_threshold` (float) - threshold value for ignoring match, default=`3.`
    * `gt_negative_threshold` (float) - threshold value for an unmatched match, default=`5.`
    * `margin` (float) - margin for the criterion, default=`null`
    * `nll_weight` (float) - weight for the proportion of NLL loss, default=`1.0`
    * `metric_weight` (float) - weight for the proportion of metric loss, default=`0.0`
    * `lr` (float) - starting learning rate, default=`1.0e-4`
    * `scheduler_gamma` (float) - value used to decay lr, default=`0.999994`
  
    * `use_cached_features` --optional-- [ONLY if features were CACHED] (bool) - flag that enables training with cached features

* `evaluation`: 
    * `epipolar_dist_threshold` (float) - threshold for epipolar distance metric, default=`5.0e-4` 
    * `camera_auc_thresholds` (List[float,...]) - thresholds for area under the curve metric, pose error in degrees, default=`[5.0, 10.0, 20.0]`  
    * `camera_auc_ransac_inliers_threshold`(float) - sampson error, default=`2.0` 


* `inference`:
  * `match_threshold` (float) - threshold for a match, default=`0.2`
    
    
* `superglue`: 
    * `descriptor_dim` (int) - dimensionality of descriptors, default=`128`
    * `laf_to_sideinfo_method` (str) - ability to include geometry info from detector for each keypoint in positional encoding, options: `[none, rotation, scale, scale_rotation, affine]`, default= `none`
    * `positional_encoding`: 
        * `hidden_layers_sizes` (List[int, ...]) - input shape for hidden layers in MLP net for positional encoding, default=`[32, 64, 128]`
        * `output_size` (int) - dimensionality of returned output, in most cases should correspond descriptor_dim, default=`128`
    * `attention_gnn`:
        * `num_stages` (int) - number of attention stages (layers), 1 stage = SELF-attn + CROSS-attn, default=`12` 
        * `num_heads` (int) - number of attention heads, default=`4` 
        * `embed_dim` (int) - corresponds to descriptor_dim, default=`128`
        * `attention` (str) - method for attention, options: `[linear, softmax]`, default=`'linear'` 
        * `use_offset` (bool) - flag for usage of offset attention https://arxiv.org/abs/2012.09688, default=`False`
    * `dustbin_score_init` (float) - dustbin score, default=`1.0` 
    * `otp`:
        * `num_iters`(int) - number of iterations for differentiable Optimal Transport solver (Sinkhorn matrix scaling algorithm), default=`20`
        * `reg` (float) - regularization value for Sinkhorn, default=`1.0`
    * `residual` (bool) - flag for enabling combining local descriptor with context-aware descriptor, default=`True` 
    
### This part is set in seperate config files from `config/features` and `config/features_online`
For each feature extractor, options: `[OPENCV_SIFT, SuperPointNet, SuperPointNetBn, OPENCVDoGAffNetHardNet]`, default=`'OPENCV_SIFT'`, this section varies, so please look in yaml files for more details.

Example of the general setup for SuperPoint case:
* `name` (str) - method name for descriptor
* `max_keypoints` (int) - maximum number of keypoints, default=`1024`
* `descriptor_dim` (int) - dimensionality of descriptors 
* `nms_kernel` (int) - size of the kernel for non-maximum suppression convolution, default=`3`
* `remove_borders_size` (int) - the number of border-neighboring pixels to skip for keypoint detection, default=`4`
* `keypoint_threshold` (float) - threshold of score confidence for keypoint to be considered, default=`0.0`
* `weights` (str) - path to the weights, option for pretrained SuperPoint weights
