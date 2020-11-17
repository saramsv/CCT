files=../data/AllNoahSaraKelleyCleanedClusters
for file_ in $files
do
    cat $file_ | grep ".png" | while read line
    do
        cluster_name=$(echo $line | cut -d ":" -f 2)

        img_name=$(echo $line | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev | sed 's/.png//')

	i=0
        grep $cluster_name $file_|  while read img
        do
            img_name2=$(echo $img | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev| sed 's/.icon//')
            echo /usb/"$img_name2"  "/data/sara/image-segmentation-keras/train_annotations/"$img_name".png" >> ../data/sup_train_seq.txt
        done
    done
done
