import argparse
import os
import albumentations as A
import cv2

def augment(training_folder, transforms_folder):
 
    # Define possible image extensions
    ext = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp', '.apng', '.avif', '.gif', '.png', '.svg', '.webp']
    
    # Iterating over all the directories inside the training directory
    for root, dirs, files in os.walk(training_folder):
        for categ in dirs:

            # Iterating over all images in each class directory
            class_path = os.path.join(root, categ)
            for image_name in os.listdir(class_path):
                im_path = os.path.join(class_path, image_name)
                if os.path.isfile(im_path) and image_name.endswith(tuple(ext)):
                    
                    # Read images for disk & convert them to RGB
                    image = cv2.imread(im_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Transform image with the augmentation pipelines we will load from yaml file
                    i=0
                    for trans_file in os.listdir(transforms_folder):
                        i+=1

                        # Load transfomers
                        transform = A.load(os.path.join(transforms_folder, trans_file), data_format='yaml')

                        # Augment image
                        transformed_image = transform(image=image)['image']

                        # Save image
                        extension = os.path.splitext(image_name)[1]
                        cv2.imwrite(class_path + '/' + image_name.split(extension)[0] + '_aug' + str(i) + extension, transformed_image)


if __name__ == '__main__':

    # Get the current path
    directory = os.getcwd()

    # Set argument parser
    parser = argparse.ArgumentParser(description="Runnnig...")
    parser.add_argument(
        "--trainning-folder", 
        "-I",

        # Set the default training directory 
        default = directory + "/dataset/training_set", 
        help = "Enter trainning images folder path"
    )
    parser.add_argument(
        "--transforms-folder", 
        "-T",

        # Set the default training directory 
        default = directory + "/transforms", 
        help = "Enter transforms folder path"
    )

    args = parser.parse_args()
    augment(args.trainning_folder, args.transforms_folder)
        
