import os
import pickle

#base_dir = "/scratch/shakeeb/benchmark_wsol_histo_tscam_bloc/exps"

#base_dir="/home/aguich25/scratch/benchmark_wsol_histo/exps/"
base_dir="/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/exps"
#methods = ["DEEPMIL"]
#datasets = ["GLAS", "CAMELYON512"]


methods = ["LayerCAM"]
datasets = ["GLAS"]


#methods = ["CAM"]
#datasets = ["GLAS"]
#da_method="ADADSA"

best_localization_score = -1
best_localization_folder = None
best_classification_score = -1
best_classification_folder = None

for method in methods:
    for dataset in datasets:
        best_localization_score = -1
        best_localization_folder = None
        best_classification_score = -1
        best_classification_folder = None

        # Construct the directory path for the current method and dataset
        #current_dir = os.path.join(base_dir, dataset, "deit_sat_tiny_patch16_224", "STD_CL",method)
        current_dir = os.path.join(base_dir, dataset, "resnet50", "NEGEV", method)
        # Walk through all directories and subdirectories in the current directory
        for root, dirs, files in os.walk(current_dir):
            # Only process directories that contain the name "target_adadsa"
            if f"glas_layercam" in root:  
                # Check if the pickle file is in the files of the current directory
                if "performance_log_best_localization.pickle" in files:
                    # Construct the full path to the pickle file
                    pickle_file = os.path.join(root, "performance_log_best_localization.pickle")
                    if os.path.exists(pickle_file):
                        # Load the pickle file
                        with open(pickle_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Get the 'best_value' for 'localization'
                        localization_best_value = data['test']['localization']['best_value']

                        # Get the 'classification'
                        classification_value = data['test']['classification']['best_value']
                        
                        # If this score is better than the current best score, update the best score and best folder
                        if localization_best_value > best_localization_score:
                            classification_score = classification_value
                            best_localization_score = localization_best_value
                            best_localization_folder = root

                    

                # Check if the pickle file is in the files of the current directory
                if "performance_log_best_classification.pickle" in files:
                    # Construct the full path to the pickle file
                    pickle_file = os.path.join(root, "performance_log_best_classification.pickle")
                    
                    if os.path.exists(pickle_file):
                        # Load the pickle file
                        with open(pickle_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Get the 'best_value' for 'classification'
                        classification_best_value = data['test']['classification']['best_value']

                        # Get the 'localization'
                        localization_value = data['test']['localization']['best_value']
                        
                        # If this score is better than the current best score, update the best score and best folder
                        if classification_best_value > best_classification_score:
                            localization_score = localization_value
                            best_classification_score = classification_best_value
                            best_classification_folder = root


        

        print(f"Classification Score: {classification_score}")
        print(f"Best Localization Score: {best_localization_score}")
        print(f"Best Folder: {best_localization_folder}")
        print("**********************************************************")

        print(f"Localization Score: {localization_score}")
        print(f"Best Classification Score: {best_classification_score}")
        print(f"Best Folder: {best_classification_folder}")
        print("**********************************************************")
