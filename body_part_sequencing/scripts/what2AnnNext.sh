file_=../data/all_clustersCleanedByNoah
cat $file_ | grep ".png" | while read line
do
cluster_name=$(echo $line | cut -d ":" -f 2)
i=0
grep $cluster_name $file_| grep -v ".png" | while read img
do
    echo $line  >> what2AnnNext
    echo $img  >> what2AnnNext
    ((i+=1))
    if [ $i == 1 ];
    then
	    break
    fi
done
done

