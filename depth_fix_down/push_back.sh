#! /bin/bash
s=`cat \$1`
lines=""
i=0
for line in $s
do
	if test $i -eq 0;then
		i=$(($i+1))
	elif test $i -eq 1;then
		color=$line
		i=$(($i+1))
	elif test $i -eq 2;then
		i=0;
		echo "object_colors.push_back($color);">>push_back.txt
	fi
done
