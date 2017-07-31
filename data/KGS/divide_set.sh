#!/bin/bash

if [ "$#" -ne 1 ] ; then
  dir=./SGFs
else
  dir=$1
fi

cd "$dir"

echo "Going to divide $dir into training, validation, and test sets."

rm -rf train ; mkdir train
rm -rf val ; mkdir val
rm -rf test ; mkdir test

file_list=sgf_list.txt
find $PWD -type f | grep '\.sgf' | shuf > "$file_list"

num_total=$(wc -l < "$file_list")
num_val=$(($num_total / 10))
num_test=$(($num_total / 10))
num_train=$(($num_total - $num_val - $num_test))

echo "Total number of minibatches is $num_mb"
echo "$num_train to training set."
echo "$num_val to validation set."
echo "$num_test to test set."

num_train_val=$(($num_train + $num_val))
conut=0
cat "$file_list" | while read file_path ; do
  (( ++count ))
  # echo "count = $count"
  if [ "$count" -le "$num_val" ] ; then
    # echo "cp $file_path val"
    cp "$file_path" val
  elif [ "$count" -le "$num_train_val" ] ; then
    # echo "cp $file_path train"
    cp "$file_path" train
  else
    # echo "cp $file_path test"
    cp "$file_path" test
  fi
done

echo "Done."
