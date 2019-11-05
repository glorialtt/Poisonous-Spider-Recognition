from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

# the path where you put your original images
your_path = '/Users/mac/Desktop/spider/Red-Headed_Mouse_Spider202'
files = os.listdir(your_path)

datagen = ImageDataGenerator(
            fill_mode='nearest')
k = 1
# for all images in the directory
for file_name in files: 
    # In macOS, .DS_Store is a hidden file
    if file_name == '.DS_Store':
        continue
    path = your_path+'/'+ file_name
    img = load_img(path, target_size = (224,224))  # this is a PIL image and resize to (224,224) per channel
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the new_path directory
    
    # save_prefix is the prefix of new images' names
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=your_path, save_prefix= os.path.splitext(filename)[0], save_format='jpg'):
        i += 1
        if i >= 1: # this will generate 10 randomly rotate images
            break  # otherwise the generator would loop indefinitely