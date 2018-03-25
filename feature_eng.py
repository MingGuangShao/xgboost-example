import numpy as np
import cPickle as pickle
import os

def cal_position(ncols,nrows,xllcorner,yllcorner,cellsize):
    # calculate the position of each point
    # output: position of each point
    grid = list()
    for i in range(int(nrows)):
        tmp = []
        for j in range(int(ncols)):
            grid_x = xllcorner + i * cellsize
            grid_y = yllcorner + j * cellsize
            tmp.append((grid_x,grid_y))
        grid.append(tmp)
    
    return grid


def cal_feature(path,name):
    #calculate the feature
    #input : filename
    #output: numpy type data

    _,data_type,date,hour = name.split('_')
    y = int(date[:4])
    m = int(date[4:6])
    d = int(date[6:])
    h = int(hour[:2])

    data = list()

    with open(path + name) as f:
        f_read = f.readlines()
        info =  [float(line.strip('\n').split(' ')[1]) for line in f_read[:6]]

        for line in f_read[6:]:
            data.append([float(i) for i in line.strip('\n').split(' ')])

        if len(data) != info[1]:
            print "read error"
        
        grid = cal_position(info[0],info[1],info[2],info[3],info[4])
        
        train_data = list()
        for i in range(int(info[1])):
            for j in range(int(info[0])):
                if data[i][j] != info[5]:#delete non_data    
                    train_data.append([y,m,d,h,grid[i][j][0],grid[i][j][1],data[i][j]])  
        #train_data = np.array(train_data)
 
        return train_data 
        
                    

if __name__ == "__main__":        
    path = "./" #define your path
    save_path = "./pickle/"
    files = os.listdir(path)
    
    #read the file in path
    for fname in files:
        if not os.path.isdir(fname) and fname[-3:] == "txt":
            print "processing file, ",fname
            data = cal_feature(path,fname)
            with open(save_path + fname[:-3] + "pickle",'wb') as fw:
                pickle.dump(np.array(data),fw)
    





