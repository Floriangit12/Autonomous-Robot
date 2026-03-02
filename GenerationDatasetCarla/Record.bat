

set "dataset=C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\CarlaUE5"
set "output_traj=%dataset%\carla_trajectories"


python RecordPath.py -n 30 -w 80  --nameFile walker_Town10_0 --maps Town10HD --sync --sample-period 0.5 --target-frames 7500 --ego-type walker --output "%output_traj%"

python RecordPath.py -n 30 -w 80  --nameFile walker_Town10_1 --maps Town10HD --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output_traj%"

python RecordPath.py -n 50 -w 60  --nameFile walker_Town10_2 --maps Town10HD --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output_traj%"

python RecordPath.py -n 50 -w 60  --nameFile walker_Town10_3 --maps Town10HD --sync --sample-period 0.5 --target-frames 7500 --ego-type walker --output "%output_traj%"



python CreateDatasetOcc3d.py --no-blur --map Town10HD --output "%dataset%\carla_occ3d_datasete\Town10_0" --trajectory-json "%output_traj%\walker_Town10_0.json" --weather-id 0 --frames 7500

python CreateDatasetOcc3d.py --no-blur --map Town10HD --output "%dataset%\carla_occ3d_datasete\Town10_1" --trajectory-json "%output_traj%\walker_Town10_1.json" --weather-id 0 --frames 10000

python CreateDatasetOcc3d.py --no-blur --map Town10HD --output "%dataset%\carla_occ3d_datasete\Town10_2" --trajectory-json "%output_traj%\walker_Town10_2.json" --weather-id 0 --frames 10000

python CreateDatasetOcc3d.py --no-blur --map Town10HD --output "%dataset%\carla_occ3d_datasete\Town10_3" --trajectory-json "%output_traj%\walker_Town10_3.json" --weather-id 0 --frames 7500

