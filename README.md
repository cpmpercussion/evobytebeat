# evobytebeat

Evolutionary Bytebeats - evolving tiny programs to create digital
audio!

Bytebeats are short computer programs can be evaulated to make a
stream of lo-fi digital audio samples! These programs are usually a
single c expression so they can be evolved using genetic programming!
Here's an example in c:

    main(t)
    {
        for(;;t++) {
            putchar( (t<<1) * 2 + (t>>2) | 21^t >> t | t>>3 );
    
        }
    }

In this project, we're exploring how evolutionary algorithms can be
used to evolve bytebeats that satisfy certain musical parameters,
e.g., pitch, bpm or volume. Can evolutionary algorithms discover some
new bytebeat styles that would be difficult to find by trial and
error?

This implementation is written in Python using the DEAP (Distributed
Evolutionary Algorithms in Python) library's genetic programming
module. Evolving bytebeats was proposed in the definitive bytebeat
paper ([Heikkilä 2011](http://arxiv.org/abs/1112.1368), see below),
but (to our knowledge) has never been attempted.

The bytebeats are formed by combining the following elements:

    t # (time step, incremented by one for each sample)
    + % * >> | & ^ - /
    1, 2, 3, 5, 7, 11, 13 # (constants)

Each bytebeat is evaluated for 40000 steps (5s at 8000kHz) and is
assigned a fitness score calculated by combining bpm and the standard
deviation of the signal. At the moment, these features are extracted
using ad hoc methods in the script, but future versions will likely
use the `aubio` library to extract further audio features.

At the end of evolution, the top bytebeats are rendered for 200000
timesteps, converted to an 8kHz 8-bit WAV file and compressed into mp3
for convenience. Each bytebeat is also visualised in a plot saved as a
`png` file.

### Why do this?

First, bytebeats sound cool; they have a lo-fi appeal
reminiscent of early computer games and experimental noise. The
bytebeat expressions directly define the output samples in a way that
makes the computer program feel like a raw creative medium, unlike
high-level symbolic music tools. This also means that the structure of
the bytebeats, and thus the process of evolution, can have a very
direct perceptual effect for the listener.

## Examples!

Here's a good one:

<iframe width="100%" height="160" src="https://clyp.it/y4i4b4dx/widget" frameborder="0"></iframe>

- [Evolved Bytebeat](https://clyp.it/y4i4b4dx)

![example byte beat](https://github.com/cpmpercussion/evobytebeat/raw/master/bytebeat-example.png)

## How to bytebeat?

0. install some stuff
  - [deap](https://github.com/deap/deap): `pip install deap`
  - [sox](http://sox.sourceforge.net): macOS: `brew install sox` ubuntu: `sudo apt-get install sox`
  - [lame](): macOS: `brew install lame`
1. run `python byte.py` to evolve and  three bytebeats which are saved
   as mp3s.
2. rock out!

## What's a bytebeat?

Short computer programs (one expression) that output samples of digital audio! Send them to a speaker for good times all around!

Ville-Matias Heikkilä (Viznut). 2011.  "Discovering novel computer music techniques by exploring the space of short computer programs" [arxiv.org/abs/1112.1368](http://arxiv.org/abs/1112.1368)

## TODO:

- experiment with including different audio features in the fitness
function (more spectral measurements would be a good next step!)
- find some way to play and display bytebeats while simultaneous
evolving them!
- so far, only the simplest EA from DEAP is used, experiment with
different options
- many more...
