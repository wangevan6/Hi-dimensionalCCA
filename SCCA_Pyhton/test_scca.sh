#!/bin/bash

# Define the ranges for a, b, and c
a_start=7
a_end=8
b_start=1
b_end=1
c1=5
c2_start=4
c2_end=4

# Define the path to the Python script
python_script="simulate.py"

# Loop through the combinations of a, b, and c
for a in $(seq $a_start $a_end); do
  for b in $(seq $b_start $b_end); do
    if [ $a -ge 0 ] && [ $a -le 4 ]; then
      c=$c1
      echo "Running with a=$a, b=$b, c=$c"
      python scca_simulation_cluster.py --a $a --b $b --c $c
    elif [ $a -ge 5 ] && [ $a -le 8 ]; then
      for c in $(seq $c2_start $c2_end); do
        echo "Running with a=$a, b=$b, c=$c"
        python scca_simulation_cluster.py --a $a --b $b --c $c
      done
    fi
  done
done

