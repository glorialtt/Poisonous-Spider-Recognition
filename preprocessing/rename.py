import os

# the path where you put your original images
your_path = '/Users/mac/Desktop/spider/dataset/test/White_Tailed_Spider'
files = os.listdir(your_path)
i = 1
for file_name in files: 
    # In macOS, .DS_Store is a hidden file
    if file_name == '.DS_Store':
        continue
    path = your_path + '/'+file_name
    new_path = your_path + '/'+ str(i) + '.' + 'White_Tailed_Spider.jpg'
    os.rename(path, new_path)
    i = i+1