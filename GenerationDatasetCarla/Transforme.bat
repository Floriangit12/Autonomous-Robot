
set "datasetUE5=C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\CarlaUE5\carla_occ3d_datasete"
set "datasetUE4=C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\CarlaUE4\carla_occ3d_datasete"


python TransformData.py --inputs %datasetUE5%\Town10_0  %datasetUE5%\Town10_1 %datasetUE5%\Town10_2 %datasetUE5%\Town10_3  %datasetUE4%\Town1_0  %datasetUE4%\Town1_1 %datasetUE4%\Town2_0 %datasetUE4%\Town2_0 %datasetUE4%\Town3_0 %datasetUE4%\Town3_1 %datasetUE4%\Town4_0 %datasetUE4%\Town4_1 %datasetUE4%\Town5_0 %datasetUE4%\Town5_1 %datasetUE4%\Town6_0 %datasetUE4%\Town6_1 %datasetUE4%\Town7_0 %datasetUE4%\Town7_0 %datasetUE4%\Town10_0 %datasetUE4%\Town10_1  --output C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\finalDataset 