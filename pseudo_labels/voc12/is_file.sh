while read line
do
	path=$(echo $line| cut -d ":" -f 1)
	if test -f $path; then
		echo $line
	fi
done < $1
