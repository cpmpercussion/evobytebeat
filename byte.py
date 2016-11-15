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


## MIR
# simple peak detection
def peak_detect(data):
    max_val = np.amax(abs(data)) 
    peak_ndx = np.where(data==max_val)
    if len(peak_ndx[0]) == 0: #if nothing found then the max must be negative
        peak_ndx = np.where(data==-max_val)
    return peak_ndx

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
    #print bpm
    return bpm,correl



def beat(t):
    return  t*(t+(t>>9|t>>13))%40&120

def playback_expr(e):
    t = 1
    while True:
        print(chr(int(
            e(t)
        ) % 256 ), end="")
        t += 1

def playback_expr_count(e):
    for t in range(50000):
        print(chr(int(
            e(t+1)
        ) % 256 ), end="")
        t += 1

def playback_char(e,t):
    return (int(e(t+1)) % 256)

#@profile
def gen_beat_output(e):
    return [playback_char(e,t) for t in range(70000)]

"""
Setup the Evolutionary Programming system
"""

def beat_division(a,b):
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

def make_random_beat():
    expr = gp.genFull(pset, min_=3,max_=10)
    tree = gp.PrimitiveTree(expr)
    #print(str(tree))
    return gp.compile(tree,pset)

def make_test_output():
    out = []
    try:
        f = make_random_beat()
        out = gen_beat_output(f)
    except:
        print("failed")
    return out

def make_test_tree():
    expr = gp.genFull(pset, min_=1,max_=3)
    tree = gp.PrimitiveTree(expr)
    return tree

def extract_feature(X):
    sample_rate = 22050
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

#@profile
def eval_beat(individual):
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

def gen_beat_stats(individual):
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
    return float(sd_score),float(bpm_score)


def output_beat_to_file(file_name, e):
    print("Writing to file:", file_name)
    routine = gp.compile(e,pset)
    with open(file_name,'w') as f:
         for t in range(200000):
             f.write(chr(int(routine(t+1))%256))

def output_beat_to_std_out(e):
    routine = gp.compile(e,pset)
    for t in range(50000):
        print((chr(int(routine(t+1))%256)), end="")

def playback_random_beat():
    a = ""
    while True:
        try:
            f = make_random_beat()
            playback_expr_count(f)
        except ValueError:
            a = "Value Error"
        except ZeroDivisionError:
            a = "Zero Div Error"
        except:
            a = "other error"
        print(a)

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
        output_beat_to_filed("individual"+str(index)+".raw",indiv) # output to files.
        # convert to wav?

def main():
    #random.seed(1024)
    #random.seed(318)
    print("Setting up Evolution of BeatBeats!")
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Starting EA Simple")
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 10, stats, halloffame=hof)
    print("Finished Evolution, now saving hall of fame.")
    for index, indiv in enumerate(hof.items):
        output_beat_to_file("best"+str(index)+".raw",indiv) # output to files!
        #output_beat_to_std_out(indiv)  # output to standard output!
    #print("Done saving hall of fame.")
    return pop, hof, stats

if __name__ == "__main__":
    main()

def sigmoid(m,x):
    return 1 / (1 + np.exp(-x))


"""
blippy, epic: (t*((15&t>>11)%12)&55-(t>>5|t>>12)|t*(t>>10)*32)-1
atmospheric, hopeful: t*3&(t>>10)|t*12&(t>>10)|t*10&((t>>8)*55)&128
expansive rumbles: t*4&(t>>10)|t*4&(t*6>>8)&t|64
electric, repetitive: t*(t+(t>>9|t>>13))%40&120
"""
