#!/bin/bash
FILE=$1

# replace the + to # chars
sed -i -r 's/^([+]{4})\s/#### /' $FILE
sed -i -r 's/^([+]{3})\s/### /' $FILE
sed -i -r 's/^([+]{2})\s/## /' $FILE
sed -i -r 's/^([+]{1})\s/# /' $FILE
sed -i -r 's/(\[php\])/<?php/' $FILE
 
# convert markdown to reStructured Text
pandoc -f markdown -t rst $FILE > ${FILE%.md}.rst

if [ -a ${FILE%.md}.md-r ] 
  then
    rm ${FILE%.md}.md-r
fi
