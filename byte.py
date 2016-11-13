"""
Evolution of ByteBeat Algorithms
Charles Martin
November 2016
"""
from __future__ import print_function
from deap import base, creator, gp, tools, algorithms
import operator
import numpy as np
import random

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
    return [playback_char(e,t) for t in range(100)]

"""
Setup the Evolutionary Programming system
"""

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mod, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.rshift,2)
pset.addPrimitive(operator.lshift,2)
pset.addPrimitive(operator.or_,2)
pset.addPrimitive(operator.and_,2)
pset.addPrimitive(operator.xor,2)
#pset.addPrimitive(operator.sub,2)
pset.addPrimitive(operator.floordiv,2)
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
    expr = gp.genFull(pset, min_=3,max_=10)
    tree = gp.PrimitiveTree(expr)
    return tree

#@profile
def evalBeat(individual):
    # compile the individual
    routine = gp.compile(individual, pset)
    # generate some test output
    try:
        test_output = gen_beat_output(routine)
    except:
        return 0.0,
    ## do some stats on the beat
    sd = np.std(np.array(test_output))
    del test_output
    # return the score
    return float(sd),

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
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox setup
toolbox.register("evaluate", evalBeat)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    #random.seed(1024)
    random.seed(318)
    print("Setting up Evolution of BeatBeats!")
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Starting EA Simple")
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 50, stats, halloffame=hof)
    #print("Finished Evolution, now saving hall of fame.")
    #for index, indiv in enumerate(hof.items):
    #    output_beat_to_file("best"+str(index)+".raw",indiv) # output to files!
        #output_beat_to_std_out(indiv)  # output to standard output!
    #print("Done saving hall of fame.")
    return pop, hof, stats

if __name__ == "__main__":
    main()




"""
blippy, epic: (t*((15&t>>11)%12)&55-(t>>5|t>>12)|t*(t>>10)*32)-1
atmospheric, hopeful: t*3&(t>>10)|t*12&(t>>10)|t*10&((t>>8)*55)&128
expansive rumbles: t*4&(t>>10)|t*4&(t*6>>8)&t|64
electric, repetitive: t*(t+(t>>9|t>>13))%40&120
"""
