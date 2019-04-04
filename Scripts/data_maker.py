import os 
from shutil import copy2
from tqdm import tqdm

root = 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\Training\\frames\\'
Tx = 50

i=1
j=0

for filename in tqdm(os.listdir(root)):
    root2 = root+filename+'\\'
    for images in tqdm(os.listdir(root2)):
        if j%Tx==0:
            path = 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\new\\'+str(i)+ '\\'
            os.mkdir('D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\new\\'+str(i))
            j=0
            i+=1
        j+=1
        src = root2+images
        dst = path
        copy2(src, dst)






