Work in Progress


cd /data001/projects/suzukiy
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv


rm -rf dirname


sbatch script.sh

scp -r root@IP:/path/to/file /path/to/filedestination


scp -r suzukiy@math-alderaan.ucdenver.pvt:/data001/projects/suzukiy/SensorStudyDeidentified/data/lstm_min_history_duration_1heart_10calories40.p   ~/code/copd/SensorStudyDeidentified/data/

scp -r suzukiy@math-alderaan.ucdenver.pvt:/data001/projects/suzukiy/SensorStudyDeidentified/data/lstm_autoencoder_min_duration_1heart_10calories40.h5   ~/code/copd/SensorStudyDeidentified/data/

scp -r "suzukiy@math-alderaan.ucdenver.pvt:/data001/projects/suzukiy/SensorStudyDeidentified/data/fitbit_clustering_data_duration_1heart_10calories40user*"   ~/code/copd/SensorStudyDeidentified/data/