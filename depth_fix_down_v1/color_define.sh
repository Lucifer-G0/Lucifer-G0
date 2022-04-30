#! /bin/bash
s=`cat \$1`
lines=""
i=0
for line in $s
do
	if test $i -eq 0;then
		color=$line
		i=$(($i+1))
	elif test $i -eq 1;then
		r=$line
		i=$(($i+1))
	elif test $i -eq 2;then
		g=$line
		i=$(($i+1))
	elif test $i -eq 3;then
		b=$line
		i=0;
		echo "#define MY_$color cv::Vec3b($r,$g,$b)">>color_define.txt
	fi
done
