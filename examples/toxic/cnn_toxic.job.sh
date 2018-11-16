#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -q r430.q
#$ -pe slot_based 8
$CL/dev/dynet/bin/x86_64__Linux__CentOS_7.2.1511__gcc7/examples/cnn_toxic/cnn_toxic
