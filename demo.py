import os
import cv2
from RDIE import *
hrImgNames = os.listdir('imgs/hr')
srImgNames = os.listdir('imgs/sr')
hrImgNames.sort(key=lambda x:x.split('.')[0])
srImgNames.sort(key=lambda x:x.split('.')[0])
hrImgNames = ['imgs/hr/'+name for name in hrImgNames]
srImgNames = ['imgs/sr/'+name for name in srImgNames]
#for calculate RDIE
metric = RDIE()
values = []
for hrName,srName in zip(hrImgNames,srImgNames):
    hr = cv2.imread(hrName)
    sr = cv2.imread(srName)
    val = metric.call(hr,sr)
    print(val)
    values.append(val)
values = np.array(values)
np.save('imgs/values.npy',values)
#for calculate and save RIE maps
RIEModel = RIE3C()
os.makedirs('imgs/RIE_hr',exist_ok=True)
os.makedirs('imgs/RIE_sr',exist_ok=True)
for hrName,srName in zip(hrImgNames,srImgNames):
    hr = cv2.imread(hrName)
    sr = cv2.imread(srName)
    hrMap = RIEModel(hr[None,:])
    srMap = RIEModel(sr[None,:])
    hrMap = 255*tf.transpose(hrMap,[3,1,2,0])[0]
    srMap = 255*tf.transpose(srMap,[3,1,2,0])[0]
    cv2.imwrite('imgs/RIE_hr/'+hrName.split('/')[-1],np.array(hrMap,np.uint8))
    cv2.imwrite('imgs/RIE_sr/'+srName.split('/')[-1],np.array(srMap,np.uint8))