# WallFollowingRobotRecognitionSystem
This includes data sets, codes, and relative document that describes results from the system. For details about motivation, how it was implemented, results, and directory, please, check the report. Since I was developing two systems, Robot Movement Recognition System, and Autism Dianosis System, there are two contents for each topic in the report. They are distinguished by (1)Wall Following Robot data set for Robot Movement Recognition System, and (2)Autism Adult data set for Autism dianosis System.

To repoduce the results in the report, you can run main_wall.py with any python IDE such as jupytor notebook, subtext, and visual studio.

***Warning***
Since there is some randomness when data sets are created in proprocessing_wall.py, you will not get the same results from the report if you run preprocessing_wall.py before you run main_wall.py.

Use preprocessing_wall.py and training_autism.py as only references.

***Tips***
If there is any error related to path, please double check you are using right paths when you run the .py files.

***note***
Data file called "sensor_readings_24_Copy1.data" couldn't be read on any IDE but Jupytor Notebook. 
You will not need to read this file in main_wall.py, but it will be required when you try to run preprocessing_wall.py. 
