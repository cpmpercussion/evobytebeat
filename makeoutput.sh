#!/bin/bash

gcc test.c
./a.out | head -c 4000000 > output.raw
sox -r 24000 -c 1 -t u8 output.raw output.wav
open output.wav
