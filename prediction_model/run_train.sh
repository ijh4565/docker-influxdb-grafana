#!/bin/bash

for j in 7 8 10 11 12
do
  echo $j
  nohup python server.py $j &

  sleep 3

  for i in 1 2 3 4 5 
  do
    nohup python client.py $j &
  done

  sleep 80
done
exit 0

