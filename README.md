# potential_automatic

This github repository let you change the potentials automatically.
Step by step guide to use

1. Install python in your computer
2. In your terminal run: pip install -r 'requirements.txt'
   2.1. If this step didn't work, create a virtual environment: python -m venv venv
   2.2. Then activate the venv:
   MAC: . venv/bin/activate
   WINDOWS: . venv/Scripts/activate
   2.3. Do the step 2. again, installing all the pip dependencies
3. Now add the raw data into the 'data' folder, you don't need to delete the old files, the python
   program will take the last added file in the folder. The file has to be a csv with minimum the following columns:
   country_code shop_id potential new_potential, with that exact names
4. create a file called 'credentials.txt' with your login and password in DiDi
5. run the 'main.py' file. It will create the .xslx files, and upload them into gattaran
