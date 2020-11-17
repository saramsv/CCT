files=../data/AllNoahSaraKelleyCleanedClusters
for file_ in $files
do
    #echo "file is "$file_
    cat $file_ | grep ".png" | while read line
    do
        #echo "line is : "$line
        cluster_name=$(echo $line | cut -d ":" -f 2)

        img_name=$(echo $line | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev | sed 's/.png//')

        #echo "clustern name: "$cluster_name", img_name: "$img_name

        #cluster=$(grep $cluster_name $file_| grep -v ".png")
	i=0
        grep $cluster_name $file_| grep -v ".png" | while read img
        do
            img_name2=$(echo $img | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev| sed 's/.icon//')
            #echo "image name2: "$img_name2
            echo "/usb/"$img_name".JPG" "/data/sara/image-segmentation-keras/train_annotations/"$img_name".png /usb/"$img_name2" /data/sara/CCT/body_part_data/data/sara_blank_img.png" >> sup_unsup.txt #>> ../data/sup_train_seq.txt 
            #echo "/usb/"$img_name2" /data/sara/CCT/body_part_data/data/sara_blank_img.png"  >> ../data/unsup_train_seq.txt
        done
    done
done
cat sup_unsup.txt | sort -u | shuf > sup_unsup_shuffed.txt
cat sup_unsup_shuffed.txt | awk -F " " '{print $1" "$2 > "sup_train_seq.txt"; print $3" "$4 >"unsup_train_seq.txt"}'

rm sup_unsup.txt
rm sup_unsup_shuffed.txt
mv sup_train_seq.txt ../data/
mv unsup_train_seq.txt ../data/
