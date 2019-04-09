import os 
from shutil import copy2
from tqdm import tqdm_notebook as tqdm
import math

root = 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\Training\\frames\\'

Tx=50;fol=0
lister = (os.listdir(root))
for l in tqdm(range(len(lister))):
    print(lister[l]+' '+str(fol))
    j=0
    root2 = os.path.join(root,lister[l])
    im_list = os.listdir(root2)
    images_in_lecture = len(im_list)
    tot_chunks = math.ceil(images_in_lecture/Tx)
    os.mkdir('D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\copied\\'+str(fol+1))
    for i in tqdm(range(tot_chunks-1)):
        while j<=images_in_lecture:
            src= os.path.join(root2, im_list[j])
            dst= 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\copied\\'+str(fol+i+1)+'\\'
            copy2(src, dst)
            j+=1
            if j%Tx==0:
                break
        os.mkdir('D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\copied\\'+str(fol+i+2))
    
    j= images_in_lecture-Tx

    while j<images_in_lecture:
        src= os.path.join(root2, im_list[j])
        dst= 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\copied\\'+str(fol+tot_chunks)
        copy2(src, dst)
        j+=1
    fol+=tot_chunks
            
        