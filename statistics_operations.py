import numpy as np
import statistics
from statistics import variance, mean

data = np.array([35,55,43,59,63,76,35,41,64,43,93,60,78,23,82])

print(data) 
print("Shape:",data.shape)  
print("Dimension:",data.ndim)
print("Data Type:",data.dtype)
print()

print("Variance:",variance(data))
print("Standard Deviation:",statistics.stdev(data))
print()

print("Mean:",statistics.mean(data))
print("Median:",statistics.median(data))
print("Mode:",statistics.mode(data))
print()

print("Mininum value:",np.min(data,axis=0)) # row
print("Maximum Value:",np.max(data,axis=0))
print()

print("Variance of the entire population/dataset:",statistics.pvariance(data))
print("Standard Deviation of the entire population:",statistics.pstdev(data))
print()

print("Central location of a data:",statistics.harmonic_mean(data))
print("Median of grouped continous data:",statistics.median_grouped(data))
print("High median of data:",statistics.median_high(data))
print("Low median of data:",statistics.median_low(data))
print()

print("20_percentile:",np.percentile(data,20))
for val in [10,20,30,50,70]:
    print(val,"_Percentile:",np.percentile(data,val))