"""
Evolution of ByteBeat Algorithms
Charles Martin
November 2016
"""
from __future__ import print_function
from deap import base, creator, gp, tools, algorithms
import operator
import numpy as np
from scipy import signal
import random
import pywt
import matplotlib.pyplot as plt
import subprocess

SOX_COMMAND = "sox -r 8000 -c 1 -t u8"
LAME_COMMAND = "lame -V 0"

## MIR
# simple peak detection
## borrowed from bpm_detection.py
## https://github.com/scaperot/the-BPM-detector-python
def peak_detect(data):
    max_val = np.amax(abs(data)) 
    peak_ndx = np.where(data==max_val)
    if len(peak_ndx[0]) == 0: #if nothing found then the max must be negative
        peak_ndx = np.where(data==-max_val)
    return peak_ndx

## borrowed from bpm_detection.py
## https://github.com/scaperot/the-BPM-detector-python
def bpm_detector(data,fs):
    cA = [] 
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2**(levels-1);
    min_ndx = 60./ 220 * (fs/max_decimation)
    max_ndx = 60./ 40 * (fs/max_decimation)
    
    for loop in range(0,levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA,cD] = pywt.dwt(data,'db4');
            cD_minlen = len(cD)/max_decimation+1;
            cD_sum = np.zeros(cD_minlen);
        else:
            [cA,cD] = pywt.dwt(cA,'db4');
        # 2) Filter
        cD = signal.lfilter([0.01],[1 -0.99],cD);

        # 4) Subtractargs.filename out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[::(2**(levels-loop-1))]);
        cD = cD - np.mean(cD);
        # 6) Recombine the signal before ACF
        #    essentially, each level I concatenate 
        #    the detail coefs (i.e. the HPF values)
        #    to the beginning of the array
        cD_sum = cD[0:cD_minlen] + cD_sum;

    if [b for b in cA if b != 0.0] == []:
        return 0.0,0.0 ## no data
    # adding in the approximate data as well...    
    cA = signal.lfilter([0.01],[1 -0.99],cA);
    cA = abs(cA);
    cA = cA - np.mean(cA);
    cD_sum = cA[0:cD_minlen] + cD_sum;
    
    # ACF
    correl = np.correlate(cD_sum,cD_sum,'full') 
    
    midpoint = len(correl) / 2
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx]);
    if len(peak_ndx) > 1:
        return 0.0,0.0 ## no data
        
    peak_ndx_adjusted = peak_ndx[0]+min_ndx;
    bpm = 60./ peak_ndx_adjusted * (fs/max_decimation)
    return bpm,correl

def playback_expr(e):
    """
    prints the byte beat as chars to standard out forever.
    """
    t = 1
    while True:
        print(chr(int(
            e(t)
        ) % 256 ), end="")
        t += 1

def playback_expr_count(e):
    """
    Prints the first 50000 chars of bytebeat to standard out.
    """
    for t in range(50000):
        print(chr(int(
            e(t+1)
        ) % 256 ), end="")
        t += 1

def playback_char(e,t):
    """
    Evaluate a bytebeat e at timestep t+1
    """
    return (int(e(t+1)) % 256)

#@profile
def gen_beat_output(e):
    """
    Returns the first 70000 steps for a bytebeat e
    """
    return [playback_char(e,t) for t in range(70000)]

"""
Setup the Evolutionary Programming system
"""
def beat_division(a,b):
    """
    Integer division protected that returns 0 for n/0.
    """
    if b == 0:
        return 0
    return a // b

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mod, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.rshift,2)
#pset.addPrimitive(operator.lshift,2)
pset.addPrimitive(operator.or_,2)
pset.addPrimitive(operator.and_,2)
pset.addPrimitive(operator.xor,2)
pset.addPrimitive(operator.sub,2)
pset.addPrimitive(beat_division,2)
pset.addTerminal(1)
pset.addTerminal(2)
pset.addTerminal(3)
pset.addTerminal(5)
pset.addTerminal(7)
pset.addTerminal(11)
pset.addTerminal(13)
pset.renameArguments(ARG0='t')

#@profile
def eval_beat(individual):
    """
    Evaluation and fitness function used for evolution.
    """
    # compile the individual
    routine = gp.compile(individual, pset)
    # generate some test output
    try:
        test_output = gen_beat_output(routine)
    except:
        return 0.0,
    ## do some stats on the beat
    sd = np.std(np.array(test_output))
    bpm, correl = bpm_detector(test_output,24000)
    bpm_score = 1 - abs((bpm/120.0)-1)
    sd_score = sd / 128.0
    del test_output
    # return the score
    return float(bpm_score * sd_score),

def output_beat_to_file(file_name, e):
    """
    Output the bytebeat to a file.
    Converts to wav with sox
    """
    print("Writing to file:", file_name)
    routine = gp.compile(e,pset)
    with open(file_name+".raw",'w') as f:
         for t in range(200000):
             f.write(chr(int(routine(t+1))%256))
    # Now convert to wav
    subprocess.call(SOX_COMMAND + " " + file_name + ".raw" + " " + file_name + ".wav", shell=True)
    subprocess.call(LAME_COMMAND + " " + file_name + ".wav", shell=True)

def output_beat_to_std_out(e):
    routine = gp.compile(e,pset)
    for t in range(50000):
        print((chr(int(routine(t+1))%256)), end="")

"""
Visualisation
"""
def bitlist(n):
    """
    Returns a list of bits for a char.
    """
    return [n >> i & 1 for i in range(7,-1,-1)]

def print_image(indiv,name):
    """
    Prints out a typical bytebeat image from an individual with a given filename.
    """
    routine = gp.compile(indiv,pset)
    output = gen_beat_output(routine)
    bits = np.array(map(bitlist,output)[0:24000]).transpose()
    plt.style.use('classic')
    plt.imshow(bits,interpolation='nearest',aspect='auto',cmap=plt.get_cmap('Greys'))
    plt.savefig(name+".png",dpi=150)

        
"""
Setup the GP evolution
"""
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=0, max_=2)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox setup
toolbox.register("evaluate", eval_beat)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def print_pop(p):
    for index, indiv in enumerate(p.items):
        output_beat_to_filed("individual"+str(index),indiv) # output to files.
        # convert to wav?

def main():
    #random.seed(1024)
    #random.seed(318)
    print("Setting up Evolution of bytebeats!")
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Starting EA Simple")
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof)
    print("Finished Evolution, now saving hall of fame.")
    for index, indiv in enumerate(hof.items):
        title = "halloffame-" + str(index)
        output_beat_to_file(title,indiv) # output to files!
        print_image(indiv,title)
        #output_beat_to_std_out(indiv)  # output to standard output!
    #print("Done saving hall of fame.")
    return pop, hof, stats

if __name__ == "__main__":
    main()
