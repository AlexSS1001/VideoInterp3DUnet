## VideoInterp3DUnet
Video interpolation network based on 3DUnet
* Step 1: download Vimeo septuplet database
```sh
wget  http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
```
* Step 2: prepare HDF5 datasets
```sh
python datasets.py --db-src-path your_vimeo_path --db-h5-path dst_hdf5_path
```
* Step 3: configure training script
* Step 4: generate HDF5 files for training and validation
* Step 5: Train with training.sh
* Step 6: Evaluate on your video of choice. Must be folder of successive frames.