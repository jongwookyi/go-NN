#!/bin/bash

if [ "$#" -ne 1 ] ; then
  echo "Must specify a directory."
  exit
fi

dir=$1
cd "$dir"

echo "Going to divide $dir into training, validation, and test sets."

file_list=./npz_list.txt

ls | grep '\.npz' | shuf > "$file_list"

num_mb=$(wc -l < "$file_list")
num_val=$(($num_mb / 10))
num_test=$(($num_mb / 10))
num_train=$(($num_mb - $num_val - $num_test))

echo "Total number of minibatches is $num_mb"
echo "$num_train to training set."
echo "$num_val to validation set."
echo "$num_test to test set."

mkdir -p train
mkdir -p val
mkdir -p test

head "-$num_val" "$file_list" | while read file_name ; do
  mv "$file_name" val
done

tail "-$num_test" "$file_list" | while read file_name ; do
  mv "$file_name" test
done

find . -name "*.npz" -exec mv {} train \;

echo "Done."
