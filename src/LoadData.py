def loadPoints(filename):
    input = open(filename, "r")
    
    info = input.readline().split()
    
    nData = int(info[0])
    nDim = int(info[1])
    
    data = [[0]*nDim for i in range(nData)]

    for i in range(nData):
        info = input.readline().split()
        for j in range(nDim):
            data[i][j] = float(info[j]) 

    return data 

def loadClusters(filename): 
    input = open(filename, "r") 
    
    info = input.readline() 
    
    nData = int(info)
    
    clusters = [0] * nData 
    
    for i in range(nData):
        info = input.readline()
        clusters[i] = int(info)
    
    return clusters

