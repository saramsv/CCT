file_=../data/all_clustersCleanedByNoah
cat $file_ | grep ".png" | while read line
do
	cluster_name=$(echo $line | cut -d ":" -f 2)

	img_name=$(echo $line | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev | sed 's/.png//')
	id_=$(echo $img_name | cut -c1-3)

	grep $cluster_name $file_|sort -u| grep -v ".png" | while read img
	do
	    img_name2=$(echo $img | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev| sed 's/.icon.JPG//')
	    #echo $img_name" : "$img_name2
	    if [ -f /home/mousavi/da1/icputrd/arf/mean.js/public/labels/$img_name2".png" ]
	    then
		    echo "/home/mousavi/da1/icputrd/arf/mean.js/public/labels/"$img_name2".png:"$cluster_name >> ../data/all_clustersCleanedByNoah_sorted
	    else
		    echo "/home/mousavi/da1/icputrd/arf/mean.js/public/sara_img/"$id_"/"$img_name2".JPG:"$cluster_name >> ../data/all_clustersCleanedByNoah_sorted
	    fi
	done
done

cat ../data/all_clustersCleanedByNoah_sorted | sort -u > ../data/all_clustersCleanedByNoah_sorted2
rm ../data/all_clustersCleanedByNoah_sorted
mv ../data/all_clustersCleanedByNoah_sorted2 ../data/all_clustersCleanedByNoah_sorted
