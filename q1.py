#!/usr/bin/env python3

from pyspark import SparkContext  # Start the pyspark

def extract_reserchers_function(record): # data --> [(sample,researcher-1),(sample,researcher-2)]
    sample, date, experiment, day, subject, kind, instrument, researchers = record.strip().split(",")
    result = [] # Save the researchers
    if sample != 'sample':
        researchers_list = researchers.strip().split('; ')
        for name in researchers_list:
            result.append((sample,name))
        return result
    else:
        return (sample,researchers) # return the title of sample and researchers
       
def extract_measurement_function(record): # return the valid measurement and 1 or 0
    data = record.strip().split(',')
    sample,FSC,SSC = data[0],data[1],data[2]
    if FSC != 'FSC-A': # Check if this is a title
        if int(FSC) >=0 and int(FSC)<=150000 and int(SSC)>=0 and int(SSC)<=150000:
            return (sample,str(1))
        else:
            return (sample,str(0))
    else:
        return (sample,str(0))

if __name__ == "__main__":
    sc = SparkContext(appName="Task 1")
    experiments = sc.textFile("/share/cytometry/experiments.csv") # Read files 'experiments'
    measurements = sc.textFile("/share/cytometry/large") # Read files 'measurements'

    # First map is extract the reseacher, flatmap is :[(sample,researcher-1),(sample,researcher-2)] --->(sample,researcher-1) \n (sample,researcher-2) 
    extract_resercher = experiments.map(extract_reserchers_function).flatMap(lambda xs: [x for x in xs]) 
    
    # extract (sample, 0)/(sample, 1), 0/1 stands for valid or not
    extract_measure = measurements.map(extract_measurement_function).repartition(1).reduceByKey(lambda before,after: int(before)+int(after))

    # Join together and add the numbers
    # (sample, researcher, sum[0,1,0,1,...]) -> (researcher, sum[0,1,0,1,...]) -> (researcher, sum[0,1,0,1,1,...])
    join_resercher_measurement = extract_resercher.join(extract_measure).values().reduceByKey(lambda before,after: int(before)+int(after)).collect()

     # sorted the data and transfer the (1,2) -->1 + '\t'+2 
    join_resercher_measurement = sc.parallelize(sorted(join_resercher_measurement,key=lambda tup:(-tup[1], tup[0]))).map(lambda record: record[0]+'\t'+str(record[1])) 
    
    join_resercher_measurement.repartition(1).saveAsTextFile("pyspark/q1") # Save it
