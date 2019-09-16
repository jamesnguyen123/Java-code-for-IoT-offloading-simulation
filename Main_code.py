# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 17:21:14 2018

@author: cdnguyen
"""
import numpy as np
import numpy.matlib
import boto3
import urllib
from scipy import signal
def lambda_handler(event, context):
    y1=[]
    y=[]
    s3 = boto3.client("s3")
    print("Hello James Nguyen")

    if event:
        print("Event : ", event)
        file_obj = event["Records"][0]
        filename = str(file_obj['s3']['object']['key'])
        filename = urllib.unquote_plus(filename)
        print("Filename: ", filename)
        fileObj = s3.get_object(Bucket = 'bucketdinh30', Key=filename)
        print("File Obj", fileObj)
        lines = fileObj["Body"].read().decode('utf8').splitlines()
        for row in lines:
            y = row.split(',')[4]
            y1.append(float(y))
            #######################
        a1=y1
        N1=len(a1)
        dim=2
        n=2
        phi=[0]*2
        fs = 5
        fc = 0.3
        # Resampling
        signal1=signal.resample(a1,int(round(len(a1)/10)))
        data1=signal1
        #Filter
        Wn = fc/(fs/2)
        dataraw = data1 - np.mean(data1)
        b,a = signal.butter(6,Wn, 'high')
        data = signal.filtfilt(b,a,dataraw)
        l=N1
        w= 2*3.14*fs/l*np.arange(-l/2, l/2)
        t2=np.fft.fft(data)
        t3=np.fft.fftshift(t2)
        t3=t3/(-w[0])
        t4=np.fft.ifft(t3)
        t5=np.fft.ifftshift(t4)
        t6=[0]*len(t5)
        for i in range(len(t5)):
            t6[i]=t5[i].real
        # Calculate Fuzzy Entropy
        series=t6
        r=0.2*np.std(series)
        N=len(series)
        for k in range(2):
            m=dim+k 
            patterns1 =[0]*m
            aux=[0]*(N-m)
            t1 =[0]*m
            y = numpy.zeros(shape=(m,N-m+1))
            for i in range(m): 
               patterns1[i]=series[i:N-m+i+1]
            patterns = np.array(patterns1) 
            for j in range(N-m+1):
                  y[:,j]=patterns[:,j]-np.mean(patterns[:,j])  
            for i in range (N-m):
               t1=np.reshape(y[:,i],(m,1))
               dist=abs(y-np.matlib.repmat(t1, 1, N-m+1))
               dist1=np.amax(dist,axis=0)
               dist2=np.exp(-1*np.power(dist1,n)/r)
               aux[i]=(np.sum(dist2)-1)/(N-m-1)
            phi[k]=sum(aux)/(N-m)
        y1=abs(phi[0])
        y2=abs(phi[1])
        FuzzyEn=np.math.log(y1)-np.math.log(y2)
        Fuzzyround=round(FuzzyEn,7)
        z=FuzzyEn
        key = str("0")
        compare = str(filename[4])
        if compare == key:
            # score
            Sara_score= 5720*np.power(z,3)-1430*np.power(z,2)+121*z-1.4
            Sara_score1=int(round(Sara_score))
            if Sara_score1 <0:
                Sara_score1=0
            if Sara_score1 >6:
                Sara_score1=6
    
            # Zero score
            Zero_score= 1570*np.power(z,3)-571*np.power(z,2)+66.3*z-1.0
            Zero_score1=int(round(Zero_score))
            if Zero_score1 <0:
                Zero_score1=0
            if Zero_score1 >2:
                Zero_score1=2
    
            # SOT score:
            SOT_score= abs(95700*np.power(z,3)-18400*np.power(z,2)+1160*z-100)
            SOT_score1=int(round(SOT_score))
            if SOT_score1 <0:
                SOT_score1=0
            if SOT_score1 >100:
                SOT_score1=100
    
            # ABC score
            ABC_score= abs(101000*np.power(z,3)-30100*np.power(z,2)+2660*z-130)
            ABC_score1=int(round(ABC_score))
            if ABC_score1 <0:
                ABC_score1=0
            if ABC_score1 >100:
                ABC_score1=100
                
            # biokin score
            Biokin_score= abs(101000*np.power(z,3)-30100*np.power(z,2)+2660*z-128)
            Biokin_score1=int(round(Biokin_score))
            if Biokin_score1 <0:
                Biokin_score1=0
            if Biokin_score1 >100:
                Biokin_score1=100
            # Berg score
            Berg_score= abs(56*np.exp(-5*z))
            Berg_score1=int(round(Berg_score))
            if Berg_score1 <0:
                Berg_score1=0
            if Berg_score1 >56:
                Berg_score1=56
            print(Berg_score1)
    
            ####################
            q="Romberg Test:" + "," + "Entropy value: "+str(Fuzzyround)+","+ "Standing Biokin score: "+str(Biokin_score1)+ "," +"Estimated Sara Score: "+str(Sara_score1)+","+"Estimated 012 Score: "+str(Zero_score1)+","+"Estimated ABC Score: "+str(ABC_score1)+","+"Estimated Berg Score: "+str(Berg_score1)
            send = s3.put_object(Bucket='bucketdinh31',Key=filename,Body=q)
        else:
            # biokin score
            Biokin_score= abs(101000*np.power(z,3)-30100*np.power(z,2)+2660*z-128)
            Biokin_score1=int(round(Biokin_score))
            if Biokin_score1 <0:
                Biokin_score1=0
            if Biokin_score1 >100:
                Biokin_score1=100
            
            # Berg score
            Berg_score= abs(56*np.exp(-5*z))*20/56
            Berg_score1=int(round(Berg_score))
            if Berg_score1 <0:
                Berg_score1=0
            if Berg_score1 >56:
                Berg_score1=56
            print(Berg_score1)
            ####################    
            q="Trunk Test:" + "," + "Entropy value: "+str(Fuzzyround)+","+ "Sitting Biokin score: "+str(Biokin_score1)+ "," +"Estimated Berg Sitting Score: "+str(Berg_score1)
            send = s3.put_object(Bucket='bucketdinh31',Key=filename,Body=q)
        
    return 'Deployment is successful'

