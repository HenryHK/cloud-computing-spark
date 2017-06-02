#!/usr/bin/env python3

import sys

import math

import numpy as np

from pyspark import SparkContext


def measurement_filter(record): # Selected the valid measurement
    data = record.strip().split(',')
    sample,FSC,SSC = data[0],data[1],data[2]
    if FSC != 'FSC-A': # If this is a tilte
        if int(FSC) >=0 and int(FSC)<=150000 and int(SSC)>=0 and int(SSC)<=150000:
            return True
        else:
            return False
    else:
        return False

def extract_measurement(record): # data ---> (sample,(Ly6C,CD11b,SCA1))
    data = record.strip().split(',')
    sample,Ly6C,CD11b,SCA1 = data[0],data[11],data[7],data[6]
    return (sample,(Ly6C,CD11b,SCA1))
    
def cluster(record): # Calculate the argmin distant
    center = np.asarray(broad_cluster_center.value)
    data = np.asarray([record[1][0],record[1][1],record[1][2]]).astype('float64')
    cluster_number = np.sum((center-data)**2,axis=1).argmin()
    return (cluster_number,(float(record[1][0]),float(record[1][1]),float(record[1][2])))

def map_result(record):
    center = np.asarray(broad_cluster_center.value) 
    data = np.asarray([record[1][0],record[1][1],record[1][2]]).astype('float64')
    cluster_number = np.sum((center-data)**2,axis=1).argmin()
    return (cluster_number+1,1)

if __name__ == "__main__":
    sc = SparkContext(appName="Task 2")
    ''' Read Data '''
    measurements = sc.textFile("/share/cytometry/large")
    after_filter_measurement = measurements.filter(measurement_filter).map(extract_measurement) # with valid data ---> (sample,(Ly6C,CD11b,SCA1))

    ''' Initial the cluster center '''
    number_of_cluster = 4
    initial_cluster = np.random.rand(number_of_cluster,3) # generate 5 center randomly
    broad_cluster_center = sc.broadcast(initial_cluster) # As broadcast
    
    ''' Cluster Learning using Forgy'''
    if len(sys.argv)>1 and sys.argv[1]>0:
        learning_time = sys.argv[1]
    else:
        learning_time = 10

    for num in range(learning_time): #Learning numbers
        cluster_ini = after_filter_measurement.map(cluster)

        # group by cluster number and re-calculate the new center (which is the center of the cluster)
        new_cluster_center = cluster_ini.groupByKey().map(lambda x : (x[0], np.sum((np.asarray(list(x[1]))),axis=0)/len(np.asarray(list(x[1])))))
        data = new_cluster_center.collect()
        data_list_tp = []
        for i in range(number_of_cluster):
            for j in data:
                if j[0] == i:
                    data_list_tp.append(j[1])
        broad_cluster_center = sc.broadcast(data_list_tp)

    ''' Finished Learning '''
    new_cluster_center_result = after_filter_measurement.map(map_result).repartition(1).reduceByKey(lambda before,after: int(before)+int(after)) # Give the data a cluster number
    number_of_cluster = new_cluster_center_result.map(lambda x : (x[0],x[1],np.asarray(broad_cluster_center.value)[x[0]-1])).sortBy(lambda record: int(record[0]))
    result = number_of_cluster.map(lambda record: str(record[0])+'\t'+str(record[1])+'\t'+str(record[2][0])+'\t'+str(record[2][1])+'\t'+str(record[2][2]))
    result.repartition(1).saveAsTextFile("pyspark/q2")
