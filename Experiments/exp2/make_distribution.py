import matplotlib.pyplot as plt
import numpy as np

'''
Step2 : 12   12   12
Step3 :  123   123
Step4 :  1234 1234
7 in total.
Each figure: probability for 1,2,3,4,N
I will gether synthesis pathways.
'''

def make_distribution(inference_datas):
    '''
    inference_datas: list of dictionaries.
        each dictionary: [smiles, true, pred, prob_tensor]
        prob_tensor: in order of (Neg, 1, 2, 3, 4)
        key: 2, 3, 4 (int type)
    '''
    color_list = [np.array([50,50,50]),np.array([100,100,233]),np.array([200,128,50]),np.array([230,30,30])]
    plt.subplots()
    pass



def make_frequency_table(datalist, n=None, start=None, end=None, norm=True):
    """function for construct frequency table of data"""
    if start == None :
        start = min(datalist)//1
    if end == None :
        end = max(datalist)//1+1
    if n == None :
        width = 1
    else :
        width = (end-start)/n

    x = [start + width*(i+0.5) for i in range(-1, n+1)]

    y = [0 for i in range(n+2)] # y[0] and y[-1] are padding. (The value of them should be 0.)

    for data in datalist :
        if start <= data < end :
            y[int((data-start)/width) + 1] += 1

    if norm :
        #To represent frequency table to distribution graph, we should normalize the table.
        norm_constant = width*sum(y)
        #return np.array(x), np.array(y)/norm_constant
        return np.array(x),np.array(y)
    return np.array(x),np.array(y)
    #return np.array(x), np.array(y)

data_set = ''

f = open('expected_value.txt')
lines = f.readlines()

data_list = {'1':[],'2':[],'3':[],'4':[]}

for line in lines:
    words = line.strip().split(' ')
    label = float(words[1])+1
    label = int(label)
    idx = int(words[3])
    if label>4 or idx > 3:
        continue
    label = str(label)
    value = float(words[2])
    value = round(value,3)
    data_list[label].append(value)

f.close()


n = len(data_set)

plot_list = ['1','2','3','4']
color_list = [np.array([50,50,50]),np.array([100,100,233]),np.array([200,128,50]),np.array([230,30,30])]
for i in range(len(color_list)):
    color_list[i] = color_list[i]/255
print (color_list)
#plot_list = ['chembl','investigation','fda']
#color_list = ['orange','purple','red']

data_plot = []

positions = list(range(1,len(plot_list)+1))

y_min = 1
y_max = 4

fig = plt.figure()
position_list = None
for name in plot_list:
    y = data_list[name]
    data_plot.append(y)

figure_plot = plt.violinplot(data_plot,positions = positions,showextrema = True,points=50)
n = len(color_list)
for i in range(n):
    violin = figure_plot['bodies'][i]
    violin.set_facecolor(color_list[i])
    violin.set_edgecolor('black')
    violin.set_alpha(1)
    bar = figure_plot['cbars']
    bar.set_color([0,0,0])

plt.show()
