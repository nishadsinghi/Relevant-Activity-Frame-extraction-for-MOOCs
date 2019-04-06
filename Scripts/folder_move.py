import os 
from shutil import move
from tqdm import tqdm

root = 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\new\\'
Tx = 5

i=1
j=0

for filename in tqdm(os.listdir(root)):
        if j%Tx==0:
            path = 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\new2\\'+str(i)+ '\\'
            os.mkdir('D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\new2\\'+str(i))
            j=0
            i+=1
        j+=1
        src = os.path.join(root, filename)
        dst = path
        move(src, dst)






