#!/bin/bash

files=($(ls *01.txt))

echo ${files[@]}

for file in ${files[@]}; do
	echo "Deleting second row from ${file}..."
	sed '2d' "${file}" > "${file%.txt}_2d.txt"
done
