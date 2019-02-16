import random
import os
import datetime
from _ast import Or, If
from time import sleep
import time
from urllib.request import urlopen
import json
import re
import requests

class DataConfig(object):
    def __init__(self, optimalData, trendFactor, leastValue, mostValue, periodReq):
        self.optimalData = optimalData
        self.trendFactor = trendFactor
        self.leastValue = leastValue
        self.mostValue= mostValue
        self.periodReq = periodReq
        
    def getOptimalData(self, nIndex):
        if len(self.optimalData) <= nIndex :
            return 0.0
        return float(self.optimalData[nIndex])
    
    def getTrendFactorData(self, nIndex):
        if len(self.trendFactor) <= nIndex:
            return "NIL"
        return self.trendFactor[nIndex]
    
    def getLeastValue(self, nIndex):
        if len(self.leastValue) <= nIndex:
            return 0
        return float(self.leastValue[nIndex])
    
    def getMostValue(self, nIndex):
        if len(self.mostValue) <= nIndex:
            return 0
        return float(self.mostValue[nIndex])
    
    def getPeriodData(self, nIndex):
        if len(self.periodReq) <= nIndex:
            return 0
        return int(self.periodReq[nIndex])

class DataTrendGen(object):
    
    def __init__(self, trendDataSet):
        self.nIndex = 0
        self.nTotal = 0
        self.IdxIncrement = True
        self.trendGenFactors = []
        trendData = trendDataSet.split(',')
        for part in trendData:
            self.trendGenFactors.append(float(part))
            self.nTotal += 1
            
    def getCount(self):
        return self.nTotal
    
    def getTrendFactorAt(self, idx):
        nValueAtIdx = 0
        if idx >= self.nTotal:
            nValueAtIdx = self.trendGenFactors[self.nTotal - 1]
        elif idx < 0:
            nValueAtIdx = self.trendGenFactors[0]
        else:
            nValueAtIdx = self.trendGenFactors[idx]
        return nValueAtIdx
        
    def getNextTrendFactor(self):
        if self.nIndex + 1 >= self.nTotal:
            self.nIndex = 0
        nValueAtIdx = self.trendGenFactors[self.nIndex]
        self.nIndex += 1 
        return nValueAtIdx

class DataPusher():
        
    def SendJsonDataToServer(self, logTime, lTLeft, lTRight, rTLeft, rTRight):
        sleep(5)
        timepoints=[];
        taperTempL1={}
        taperTempL1["name"] = "LeftTaperLeftTemp"
        taperTempL1["value"] = lTLeft
        taperTempL1["timestamp"] = int(time.time()*1000)
        taperTempL1["tags"]={}
        taperTempL1["tags"]["banking"]="True"
        taperTempL1["tags"]["status"]="OPEN"
        timepoints.append(taperTempL1)
        
        taperTempL2={}
        taperTempL2["name"] = "LeftTaperRightTemp"
        taperTempL2["value"] = lTRight
        taperTempL2["timestamp"] = int(time.time()*1000)
        taperTempL2["tags"]={}
        taperTempL2["tags"]["banking"]="True"
        taperTempL2["tags"]["status"]="OPEN"
        timepoints.append(taperTempL2)
        
        taperTempR1={}
        taperTempR1["name"] = "RightTaperLeftTemp"
        taperTempR1["value"] = rTLeft
        taperTempR1["timestamp"] = int(time.time()*1000)
        taperTempR1["tags"]={}
        taperTempR1["tags"]["banking"]="True"
        taperTempR1["tags"]["status"]="OPEN"
        timepoints.append(taperTempR1)
        
        taperTempR2={}
        taperTempR2["name"] = "RightTaperRightTemp"
        taperTempR2["value"] = rTRight
        taperTempR2["timestamp"] = int(time.time()*1000)
        taperTempR2["tags"]={}
        taperTempR2["tags"]["banking"]="True"
        taperTempR2["tags"]["status"]="OPEN"
        timepoints.append(taperTempR2)
        
        taperTempRul={}
        taperTempRul["name"] = "RUL"
        taperTempRul["value"] = 3
        taperTempRul["timestamp"] = int(time.time()*1000)
        taperTempRul["tags"]={}
        taperTempRul["tags"]["banking"]="True"
        taperTempRul["tags"]["status"]="OPEN"
        timepoints.append(taperTempRul)
        
        print ("Bulk ready.");
        print (json.dumps(timepoints));
        r = requests.post("http://localhost:8083/api/v1/datapoints", data=json.dumps(timepoints))
        #r = requests.post("http://kairosdb:8083/api/v1/datapoints", data={"TimeStamp":int(time.time()*1000),"LeftTaperLeftTemp":lTLeft,"LeftTaperRightTemp":lTRight,"RightTaperLeftTemp":rTLeft,"RightTaperRightTemp":rTRight})
        print(r.status_code, r.reason)
        print ("Bulk gone.")
        
class BigDataCreator:
    
    def __init__(self):
        self.dataPusher = DataPusher
        
    def CreateOutputFile(self):
        flePath = "../../KerasFeasibility/Data/DataToTeach.csv"
        self.CreateOutputDirectory(flePath)
        return open(flePath, "w+") 
    
    def CreateOutputDirectory(self, flePath):
        directory = os.path.dirname(flePath)
        if not os.path.exists(directory):
            os.mkdir(directory)
        
    def WriteLine(self, fle, line):
        #print (line)
        fle.write(line)

    def ReadBlockData(self, fileName):
        blockData = []
        file = open(fileName, 'r')
        for line in file:
            parts = line.split('\t')
            blockPart = []
            for part in parts:
                blockPart.append(float(part))
            blockData.append(blockPart)
        return blockData

    def ReadBlockMedian(self, fileName):
        data = []
        file = open(fileName, 'r')
        for line in file:
            data.append(float(line))
        return data

    def GetTrendData(self, trendDataSet):
        return DataTrendGen(trendDataSet)

    def CreateNextSetOfData(self, fle, optimalData, trendFactorSet, leastValue, mostValue, monthForHalt, sendLiveData):
        nUpperMedian = 5
        nLowerMedian = -5
        ntrendVar = 0;
        haltProduction = False
        nextStepFreq = 0
        
        # How may data to send in the specified months 
        periodReq = monthForHalt *  self.getFrequencyPerMonth()
        
        #RUL value
        rulLowValueRanges = []
        rulHighValueRanges = []
        rulRemMonthsWRTRange = []
        
        midVal = (mostValue + leastValue) / 2.0
        dissectionFactor = (midVal - leastValue) / monthForHalt
        valueRange = midVal - dissectionFactor
        iteration = 0
        
        while valueRange > leastValue:
            rulLowValueRanges.append(valueRange)
            valueRange -= dissectionFactor
            rulRemMonthsWRTRange.append(monthForHalt - iteration)
            iteration += 1
        rulLowValueRanges.append(leastValue)
        rulRemMonthsWRTRange.append(monthForHalt - iteration)
            
        valueRange = midVal + dissectionFactor
        while valueRange < mostValue:
            rulHighValueRanges.append(valueRange)
            valueRange += dissectionFactor
        rulHighValueRanges.append(mostValue)
        
        trendFactors = self.GetTrendData(trendFactorSet)
        nextStepFreq = int(periodReq / trendFactors.getCount())

        for timePeriod in range(periodReq):

            if 0 == timePeriod % (nextStepFreq / 2):
                self.LogTime += 1
            
            if 0 == timePeriod % nextStepFreq:
                ntrendVar = trendFactors.getNextTrendFactor();
                
            ltLeft = optimalData + ntrendVar + random.randint(nLowerMedian, nUpperMedian)
            ltRight = optimalData + ntrendVar + random.randint(nLowerMedian, nUpperMedian)
            rtLeft = optimalData + ntrendVar + random.randint(nLowerMedian, nUpperMedian)
            rtRight = optimalData + ntrendVar + random.randint(nLowerMedian, nUpperMedian)
            
            if timePeriod < periodReq - 1:
                if ltLeft <= leastValue:
                    ltLeft = leastValue + random.randint(nLowerMedian, nUpperMedian)
                if ltRight <= leastValue:
                    ltRight = leastValue + random.randint(nLowerMedian, nUpperMedian)
                if rtLeft <= leastValue:
                    rtLeft = leastValue + random.randint(nLowerMedian, nUpperMedian)
                if rtRight <= leastValue:
                    rtRight = leastValue + random.randint(nLowerMedian, nUpperMedian)
                if ltLeft >= mostValue:
                    ltLeft = mostValue - random.randint(nLowerMedian, nUpperMedian)
                if ltRight >= mostValue:
                    ltRight = mostValue - random.randint(nLowerMedian, nUpperMedian)
                if rtLeft >= mostValue:
                    rtLeft = mostValue - random.randint(nLowerMedian, nUpperMedian)
                if rtRight >= mostValue:
                    rtRight = mostValue - random.randint(nLowerMedian, nUpperMedian)
            else:
                if (ltLeft < leastValue or ltRight < leastValue or \
                    rtLeft < leastValue or rtRight < leastValue or \
                    ltLeft > mostValue or ltRight > mostValue or \
                    rtLeft > mostValue or rtRight > mostValue):
                    haltProduction = True
                else:
                    upTrend = True
                    firstHalfSum = 0.0
                    secondHalfSum = 0.0
                    for counter in range(trendFactors.getCount()):
                        if counter > trendFactors.getCount() / 2:
                            secondHalfSum += trendFactors.getTrendFactorAt(counter) 
                        else:
                            firstHalfSum += trendFactors.getTrendFactorAt(counter)                              

                    if firstHalfSum > secondHalfSum:
                        upTrend = False
                        
                    listFourValues = [ltLeft, ltRight, rtLeft, rtRight]
                    
                    if upTrend == True:
                        bigValue = max(listFourValues)
                        if ltLeft == bigValue:
                            ltLeft = mostValue + 0.1
                        elif ltRight == bigValue:
                            ltRight = mostValue + 0.1
                        elif rtLeft == bigValue:
                            rtLeft = mostValue + 0.1
                        else:
                            rtRight = mostValue + 0.1
                    else:
                        lowValue = min(listFourValues)
                        if ltLeft == lowValue:
                            ltLeft = leastValue - 0.1
                        elif ltRight == lowValue:
                            ltRight = leastValue - 0.1
                        elif rtLeft == lowValue:
                            rtLeft = leastValue - 0.1
                        else:
                            rtRight = leastValue - 0.1
                    haltProduction = True
                    
            rulMonthsLTForLowRange = 0
            rulMonthsRTForLowRange = 0
            if (ltLeft < leastValue or ltRight < leastValue or \
                rtLeft < leastValue or rtRight < leastValue or \
                ltLeft > mostValue or ltRight > mostValue or \
                rtLeft > mostValue or rtRight > mostValue):
                haltProduction = True
            else:
                iter = 1
                rulMonthsForltLeft = rulRemMonthsWRTRange[0]
                rulMonthsForltRight = rulRemMonthsWRTRange[0]
                for lowVal in rulLowValueRanges:
                    if ltLeft < lowVal:
                        rulMonthsForltLeft = rulRemMonthsWRTRange[iter]
                    if ltRight < lowVal:
                        rulMonthsForltRight = rulRemMonthsWRTRange[iter]
                    iter += 1
                if rulMonthsForltLeft < rulMonthsForltRight:
                    rulMonthsLTForLowRange = rulMonthsForltLeft
                else:
                    rulMonthsLTForLowRange = rulMonthsForltRight
    
                iter = 1
                rulMonthsForrtLeft = rulRemMonthsWRTRange[0]
                rulMonthsForrtRight = rulRemMonthsWRTRange[0]
                for highVal in rulHighValueRanges:
                    if rtLeft > highVal:
                        rulMonthsForrtLeft = rulRemMonthsWRTRange[iter]
                    if rtRight > highVal:
                        rulMonthsForrtRight = rulRemMonthsWRTRange[iter]
                    iter += 1
                if rulMonthsForrtLeft < rulMonthsForrtRight:
                    rulMonthsRTForLowRange = rulMonthsForrtLeft
                else:
                    rulMonthsRTForLowRange = rulMonthsForrtRight
                
            rulMonthsForSystem = 0
            if rulMonthsLTForLowRange < rulMonthsRTForLowRange:
                rulMonthsForSystem = rulMonthsLTForLowRange
            else:
                rulMonthsForSystem = rulMonthsRTForLowRange
                
            if sendLiveData == True:
                self.dataPusher.SendJsonDataToServer(self, logTime=self.LogTime, lTLeft=ltLeft, lTRight=ltRight, rTLeft=rtLeft, rTRight=rtRight)
            else:
                #self.startTimeValue += datetime.timedelta(seconds=30)
                #time = self.startTimeValue
                line = '{0},{1},{2},{3},{4},{5}\n'.format(self.cycleID, self.LogTime, ltLeft, ltRight, rtLeft, rtRight)
                self.WriteLine(fle, line)

            if haltProduction == True:
                break

    def getFrequencyPerMonth(self):
        #return 2 * 60 * 24 * 30
        return 2 * 30

    def ReadDataConfig(self, fileName):
        optimalData = []
        trendFactor = []
        leastValue  = []
        mostValue   = []
        periodReq   = []
        file = open(fileName, 'r')
        for line in file:
            parts = line.split(':')
            if parts[0][0] == '#':
                continue
            optimalData.append(parts[0])
            trendFactor.append(parts[1])
            leastValue.append(parts[2])
            mostValue.append(parts[3])
            periodReq.append(parts[4])
            #data.append(float(line))
        return DataConfig(optimalData, trendFactor, leastValue, mostValue, periodReq)
    
    def Create(self, sendLiveData):
        
        dataCnfg = self.ReadDataConfig('../InitialData/DataConfig.txt')
        fle = self.CreateOutputFile()
        
        #self.startTimeValue = datetime.datetime(2018, 1, 17, 21, 47, 13, 90244)
        self.cycleID = 0
        self.LogTime = 0;
        nIdxData = 0
        optimumData = dataCnfg.getOptimalData(nIdxData)        
        while 0.0 < optimumData:
            self.cycleID += 1
            self.LogTime = 0;
            trendFactor = dataCnfg.getTrendFactorData(nIdxData)
            monthForHalt = int(dataCnfg.getPeriodData(nIdxData))
            leastVal = dataCnfg.getLeastValue(nIdxData)
            mostVal= dataCnfg.getMostValue(nIdxData)
            self.CreateNextSetOfData(fle, optimumData, trendFactor, leastVal, mostVal, monthForHalt, sendLiveData)
            nIdxData += 1
            optimumData = float(dataCnfg.getOptimalData(nIdxData))

#starttimeValue = datetime.datetime(100,1,1,11,34,59) #datetime.datetime(2018, 1, 17, 21, 47, 13, 90244)
creator = BigDataCreator()
creator.Create(True)
