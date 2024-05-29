import os
import random
import shutil
from itertools import islice

outputfolderpath="dataset/splitdata"
inputfolderpath="dataset/all"
splitratio={"train":0.7,"val":0.2,"test":0.1}
classes=["fake","real"]

try:
    shutil.rmtree(outputfolderpath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputfolderpath)

os.makedirs(f"{outputfolderpath}/train/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/train/labels",exist_ok=True)
os.makedirs(f"{outputfolderpath}/val/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/val/labels",exist_ok=True)
os.makedirs(f"{outputfolderpath}/test/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/test/labels",exist_ok=True)

listnames=os.listdir(inputfolderpath)
uniquenames=[]
for name in listnames:
    uniquenames.append(name.split('.')[0])
uniquenames=list(set(uniquenames))

random.shuffle(uniquenames)

lendata=len(uniquenames)
lentrain=int(lendata*splitratio['train'])
lenval=int(lendata*splitratio['val'])
lentest=int(lendata*splitratio['test'])

if lendata!=lentrain+lentest+lenval:
    remaining=lendata-(lentrain+lentest+lenval)
    lentrain+=remaining

lengthToSplit=[lentrain,lenval,lentest]
input=iter(uniquenames)
output = [list(islice(input, elem)) for elem in lengthToSplit]


sequence = ['train', 'val', 'test']
for i,out in enumerate(output):
    for fileName in out:
        shutil.copy(f'{inputfolderpath}/{fileName}.jpg', f'{outputfolderpath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputfolderpath}/{fileName}.txt', f'{outputfolderpath}/{sequence[i]}/labels/{fileName}.txt')

datayaml=f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f=open(f"{outputfolderpath}/data.yaml",'a')
f.write(datayaml)
f.close()