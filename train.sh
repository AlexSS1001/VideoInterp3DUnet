# Create hdf5 versions
python datasets.py --db-src-path /mnt/f/Projects/video_interp_pl/DB/vimeo_septuplet --db-h5-path /mnt/d/vimeo_septuplet_h5
# train based on config
python train.py
