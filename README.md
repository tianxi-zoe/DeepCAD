# DeepCAD Use Manual

DeepCAD denoising method was used and adjusted based on the original base model [DeepCAD](https://github.com/cabooster/DeepCAD) developed by XinYang Li et al,. For training purposes on new datatypes, it requires GPUs to accelerate the training procedure. The training process was completed on [Digital Research Alliance of Canada](https://alliancecan.ca/en) (DRAC). For testing purposes, it can be done locally on personal laptops. There is an existing trained model based on calcium flicker data using Human Umbilical Vein Endothelial Cells (HUVECs) in the **master** branch.

## 1. Set up environment in DRAC

1. Register an account with DRAC. The account needs to be renewed each year. 
2. Log into your account using `ssh -Y username@machine_name`. You can also set up the remote account using the IDE of your choice. For details, please refer to this [page](https://docs.alliancecan.ca/wiki/SSH). 
3. Create the virtual environment and install python as [described](https://docs.alliancecan.ca/wiki/Python). The python required by DeepCAD is Python 3.9 but Python 3.8 also works. Change ENV to the environment name you'd like to use.
	```[name@server ~]$ virtualenv --no-download ENV

	 [name@server ~]$ source ENV/bin/activate

	 (ENV)[name@server ~]$ module avail python

	 (ENV)[name@server ~]$ module load python/3.8

	 (ENV)[name@server ~]$ pip install --no-index --upgrade pip```
5. To request a job using CPU or GPU resouces, you can do
     - `salloc --time=2:30:0 --ntasks=1 --gres=gpu:1 --cpus-per-task=6 --mem-per-cpu=8000M --account=def-someuser`
     - submit a `.sh` job script, refer to `train.sh` or [this page](https://docs.alliancecan.ca/wiki/Running_jobs#GPU_job). 
     >To check the status of current active jobs, run the commend `sq`. 
     To cancel a job, run the command `scancel <jobid>`.
     To check the disk usage, run the command `diskusage_report`.
## 2. Set up DeepCAD in your virtual environment
1. Clone the **master ** branch into your local or cloud path
	`git clone -b master https://github.com/tianxi-zoe/DeepCAD.git` 
	`cd DeepCAD`
2. Install all the packages required
	`pip install -r requirements.txt`
	
## 3. Test new calcium flicker movies with exsiting model
1. Upload the testing data to the cloud if you are using DRAC using *VS code*, *Filezilla*, *Cyberduck* or other transfer tools. 
2. Activate the environment.
	`source ENV/bin/activate`
4. Change the parameters as needed in *demo_test_pipeline.py* such as 
    - datasets_path: the absolute/relative path to the testing dataset
    - denoise_model: the denoising model to use
    - test_datasize: the number of output frames, increase or decrease it as needed.
    - GPU: i.e. '0', '0,1'. Use '0' if only one GPU is used.
    -  output_dir: the output directory for the denoised results
 3. Run the script
	 `python3 demo_test_pipeline.py`

## 4. Train a new model with new types of datasets

1. **Training with GPUS is highly recommended**. Upload the testing data to the cloud if you are using DRAC using *VS code*, *Filezilla*, *Cyberduck* or other transfer tools. 
2. Activate the environment.
	`source ENV/bin/activate`
3. Change the parameters as needed in *demo_train_pipeline.py* such as 
    - datasets_path: the absolute/relative path to the testing dataset
    - n_epochs: number of iterations over the whole datasets 
    - GPU: i.e. '0', '0,1'. Use '0' if only one GPU is used.
 4. Run the script
	 `python3 demo_train_pipeline.py`
 5. It may take hours or days for the training to complete depending on your dataset size and your GPU model. The model of each epoch is saved in `./pth`, which can be used to test new data similar to training data. 
 6. If you are using DRAC, you can also submit a `.sh` task so that it can run from the back end. Refer to `train.sh`.

# Quantification of calcium flickers in the video
- Automated camera noise correction was done for all raw videos using `process_raw_data.py`.
- Automated counting of calcium flickers was developed. `snr_deepcad.py` detects calcium flickers present in denoised videos and outputs a *json* file containing location, duration, area, and average intensity. Modify the `threshold, min_size, max_size, min_duration and max_duration` to meet with other imaging conditions. It also outputs the signal-to-noise ratio (SNR) of each video with each flicker as a separate entry in the *json* file. 
- The violin plots of duration and area were generated using `area_violin_plot.py`. Significance was evaluated with t-test.
- The scatter plot of intensity vs duration was generated using `duration_intensity_scatter_plot.py`.
- Pseudo High Exposure time videos were generated using `pseudo.py`. Adjust the number of frames to add up to achieve different pseudo exposure time. 
