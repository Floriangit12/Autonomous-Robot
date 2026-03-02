Create path : 
python RecordPath.py --maps Town01=4096,Town02=4096,Town03=4096,Town04=4096,Town05=4096,Town06=4096,Town07=4096,Town10HD_Opt=4096

Create occupancy 3d: 
python CreateDatasetOcc3d.py  --no-blur   --frames 4096 --map Town01  --output CarlaUE4\carla_occ3d_datasetee\Town1_mod0 --trajectory-json CarlaUE4\carla_trajectories\Town01_walker_20251018_174916_ego895.json --weather-id 0

    


python TransformData.py --inputs CarlaUE4/carla_occ3d_datasete/Town1_mod0 CarlaUE4/carla_occ3d_datasete/Town1_mod2 CarlaUE4/carla_occ3d_datasete/Town2_mod0 CarlaUE4/carla_occ3d_datasete/Town2_mod2 CarlaUE4/carla_occ3d_datasete/Town3_mod0 CarlaUE4/carla_occ3d_datasete/Town3_mod2 CarlaUE4/carla_occ3d_datasete/Town4_mod0 CarlaUE4/carla_occ3d_datasete/Town4_mod2 CarlaUE4/carla_occ3d_datasete/Town5_mod0 CarlaUE4/carla_occ3d_datasete/Town5_mod2 CarlaUE4/carla_occ3d_datasete/Town6_mod0 CarlaUE4/carla_occ3d_datasete/Town6_mod2 CarlaUE4/carla_occ3d_datasete/Town10_mod2 CarlaUE4/carla_occ3d_datasete/Town10_mod2 --output CarlaUE4/carla_occ3d_datasete/mergeData --seed 42


python TransformData.py --inputs DataSet/carla_occ3d_datasete/Town10_0 DataSet/carla_occ3d_datasete/Town10_1 DataSet/carla_occ3d_datasete/Town10_2 DataSet/carla_occ3d_datasete/Town10_3 DataSet/carla_occ3d_datasete/Town10_4  DataSet/carla_occ3d_datasete/Town10_5  --output DataSet/carla_occ3d_datasete/finalDataset/40k --seed 42


C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\carla_occ3d_datasete\Town10_2




python RecordPath.py --maps Town01=7500,Town02=7500,Town03=7500,Town04=10000,Town05=7500,Town06=5000,Town07=5000,Town10HD_Opt=20000