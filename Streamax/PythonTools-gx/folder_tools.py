import os
import os.path

test_path = 'G:/Train_Data/new_dsm_image'

def RenameFolderFiles(folder_path, rename, pos=0):
    print(folder_path + ' start!')
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(file_path):
            print(file + 'is not exists!')
        os.rename(file_path, os.path.join(folder_path, rename + file))

    print(folder_path+' done!')
    return

if __name__ == '__main__':
    for folder in os.listdir(test_path):
        smoking_folder_path = os.path.join(test_path,folder+'/smoking')
        if os.path.exists(smoking_folder_path):
            RenameFolderFiles(smoking_folder_path,folder)