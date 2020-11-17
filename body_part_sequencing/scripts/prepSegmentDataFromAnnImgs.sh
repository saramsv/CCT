
#cat 8donors_annotated_imgs | awk -F "," '{print $( NF -3 )}'| cut -d '/' -f2| sort -u > annotated_donor_ids
#cat annotated_donor_ids | while read id
#do
#    cat 8donors_annotated_imgs | grep "sara_img/"$id | awk -F "," '{print $( NF -3 )}' |sort -u |  while read img_name
#    do
#        img_name=$(echo $img_name | sed 's/JPG/icon.JPG/')
#        corresponding_cluster_name=$(grep $img_name sequences/$id"_pcaed_seuenced" | cut -d ":" -f 2)
#        #corresponding_cluster=$(grep $corresponding_cluster_name$ sequences/$id"_pcaed_seuenced") #| grep -v $img_name
#        echo $corresponding_cluster_name
#        grep -w $corresponding_cluster_name$ sequences/$id"_pcaed_seuenced" | while read img
#        do
#            if [[ "$img" == *"$img_name"* ]];
#            then
#                echo "/home/mousavi/da1/icputrd/arf/mean.js/public/labels_nobg/"$(echo $img_name | cut -d '/' -f 3|sed 's/icon.JPG/png/')": "$corresponding_cluster_name
#            fi
#            echo $img
#        done
#    done
#    break
#done





########## for annotaed sequences

ls  /data/sara/image-segmentation-keras/train_annotations/|  while read img_name
do
	img_name=$(echo $img_name | sed 's/.png//')
	id=$(echo $img_name | cut -c1-3)
	if [ -f ../data/sequences/$id"_pcaed_sequenced" ]
	then
		corresponding_cluster_name=$(grep $img_name ../data/sequences/$id"_pcaed_sequenced" | cut -d ":" -f 2)
		i=0
		echo "/home/mousavi/da1/icputrd/arf/mean.js/public/labels/"$img_name".png: "$corresponding_cluster_name >> ../data/AllClusters2beCleanedByNoahRound2
		if [ "$corresponding_cluster_name" != "shade" ] | [ "$corresponding_cluster_name" != "plastic" ] | [ "$corresponding_cluster_name" != "stake" ];
		then 
			grep -w $corresponding_cluster_name$ ../data/sequences/$id"_pcaed_sequenced" | while read img #&& [ $i -le 5 ]
			do
			#echo "/data/sara/Imgs/"$img_name".JPG" "/data/sara/CCT/body_part_data/labels/"$img_name".png" >> ../data/sup_train_seq.txt 
			#echo "/data/sara/Imgs/"$(echo $img | cut -d '/' -f 11|sed 's/icon.JPG.*/JPG/')  /data/sara/CCT/body_part_data/data/sara_blank_img.png >> ../data/unsup_train_seq.txt
			echo $img >> ../data/AllClusters2beCleanedByNoahRound2
			#((i+=1))
			done
		fi
	fi
done


################# for val
#ls /data/sara/CCT/body_part_data/val_imgs/ |  while read img_name
#do
#	img_name=$(echo $img_name | sed 's/.png//')
#    echo "/data/sara/Imgs/"$img_name".JPG" "/data/sara/CCT/body_part_data/val_imgs/"$img_name".png" >> ../data/val.txt 
#done
