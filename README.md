# CSMCSCI575GROUP
## GIT Documentation
* git pull: Update the current branch. Usually you want to do this from Master. 
* git checkout <name of branch>: change which branch you are on
* git status: see what changes you made
* git add <name of file to add>: I want to save this file
* git commit -m "your message": add a comment to your changes
* git push origin <name of branch>: where you want to save your changes on the online repository

## How to Run Our GUI (FFNN or CNN)
* cd into src/gui and run the command python user_interface.py
* if you have uninstalled libraries, please install them and try running the script again
* once the GUI window pops-up, click on the browse for image button
* this will open a finder/ file explorer window, you may only select .jpg or .png files
* next, select either FFNN or CNN from the algorthim choices (the script will take a few seconds to load the image into the model)
* the output predicted mask will be saved to a file called Result.png and should display on the right of the GUI
* toggle between the model's predicted mask and the input image with the view mask button

## How to Run a Custom Algorithm
* for your algorithm scirpt to be integrated into our GUI, you will need to make sure it takes a filename to the .png or .jpg as an input and saves its predicted mask to a file called "Results.png". You will also need to ensure that the script is able to access the corresponding model.h file.
* after you meet this requirements, you will be able to select your .py file by clicking the browse models button
* click on the Custom algorithm button to execute your own script.
