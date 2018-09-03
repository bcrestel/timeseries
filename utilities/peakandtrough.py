#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:00:02 2018

@author: ben
"""
import pandas as pd
from ML.datasets.timeseries.read_stocks import loadfile, compounded


def findpeakandtrough(df, ratio_downturn=0.1, ratio_recovery=0.1):
    """ function to returns peaks and throughs of a time-series"""    
    downturn = False
    peakandtrough = []
    currmax = [df.index[0], df.iloc[0]]
    
    for ii in range(len(df)):
        dat = df.index[ii]
        val = df.iloc[ii]
        if not downturn:
            if val > currmax[1]:
                currmax = [dat, val]
            if val < (1.0-ratio_downturn) * currmax[1]:
                downturn = True
                currpth = [currmax[0]]
                currmin = [dat, val]
        else:
            if val < currmin[1]:
                currmin = [dat, val]
            if val > (1.0+ratio_recovery)*currmin[1]:
                downturn = False
                currpth.append(currmin[0])
                peakandtrough.append(currpth)
                currpth = []
                currmax = [dat, val]
    if len(currpth) > 0:
        peakandtrough.append(currpth)
    return peakandtrough


if __name__ == "__main__":
    df = loadfile('../datasets/timeseries/stocksIBMandCo.txt')
    df['MktC'] = compounded(df.MARKET) + 1.0
    #df = df.iloc[90:,:]
    df.set_index('date').MktC.plot()
    pandt = findpeakandtrough(df.set_index('date').MktC)
    print(pandt)