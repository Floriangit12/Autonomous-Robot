
set "output=C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\CarlaUE4\carla_trajectories"


@REM python RecordPath.py -n 30 -w 70  --nameFile vehicle_Town01 --maps Town01 --sync --sample-period 0.5 --target-frames 10000 --ego-type vehicle --output "%output%"

@REM python RecordPath.py -n 30 -w 70  --nameFile walker_Town01 --maps Town01 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


@REM python RecordPath.py -n 30 -w 70  --nameFile vehicle_Town02 --maps Town02 --sync --sample-period 0.5 --target-frames 10000 --ego-type vehicle --output "%output%"

python RecordPath.py -n 40 -w 70  --nameFile walker_Town02 --maps Town02 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


python RecordPath.py -n 30 -w 70  --nameFile vehicle_Town03 --maps Town03 --sync --sample-period 0.5 --target-frames 10000 --ego-type vehicle --output "%output%"

python RecordPath.py -n 20 -w 70  --nameFile walker_Town03 --maps Town03 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


python RecordPath.py -n 30 -w 70  --nameFile vehicle_Town04 --maps Town04 --sync --sample-period 0.5 --target-frames 10000 --ego-type vehicle --output "%output%"

python RecordPath.py -n 40 -w 70  --nameFile walker_Town04 --maps Town04 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


python RecordPath.py -n 20 -w 70  --nameFile walker_Town05 --maps Town05 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"

python RecordPath.py -n 40 -w 70  --nameFile walker_Town05 --maps Town05 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


python RecordPath.py -n 30 -w 70  --nameFile vehicle_Town06 --maps Town06 --sync --sample-period 0.5 --target-frames 10000 --ego-type vehicle --output "%output%"

python RecordPath.py -n 40 -w 70  --nameFile walker_Town06 --maps Town06 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


python RecordPath.py -n 30 -w 70  --nameFile vehicle_Town07 --maps Town07 --sync --sample-period 0.5 --target-frames 10000 --ego-type vehicle --output "%output%"

python RecordPath.py -n 40 -w 70  --nameFile walker_Town07 --maps Town07 --sync --sample-period 0.5 --target-frames 10000 --ego-type walker --output "%output%"


python RecordPath.py -n 50 -w 70  --nameFile vehicle_Town10 --maps Town10HD --sync --sample-period 0.5 --target-frames 15000 --ego-type vehicle --output "%output%"

python RecordPath.py -n 60 -w 70  --nameFile walker_Town10 --maps Town10HD --sync --sample-period 0.5 --target-frames 15000 --ego-type walker --output "%output%"

