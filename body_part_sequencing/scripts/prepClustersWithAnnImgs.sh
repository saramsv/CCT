i=0
ls /data/sara/CCT/body_part_data/labels/ |  while read img_name
do
	img_name=$(echo $img_name | sed 's/.png//')
	id=$(echo $img_name | cut -c1-3)
	cluster_name=$(grep $img_name ../data/sequences/$id"_pcaed_sequenced" | cut -d ":" -f 2)
    if [ "$cluster_name" == "shade" ] | [ "$cluster_name" == "plastic" ] | [ "$cluster_name" == "stake" ] | [ -z "$cluster_name" ];
    then 
        continue
    else
        n=$(($i%30))
	    echo "/home/mousavi/da1/icputrd/arf/mean.js/public/labels/"$img_name".png: "$cluster_name >> ../data/clusters_with_annotations$n
        ((i+=1))
        grep -w $cluster_name$ ../data/sequences/$id"_pcaed_sequenced" | while read img
        do
            echo $img >> ../data/clusters_with_annotations$n
        done
    fi
    #if [ $i == 5 ];
    #then
    #    break
    #fi
done


