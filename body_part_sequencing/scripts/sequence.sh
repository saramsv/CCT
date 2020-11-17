cat ../data/bodyPartAnntags.csv.20201027FixedLabels | awk -F ',' '{print $(NF -4)}' > ../data/annotated_imgs
cat ../data/annotated_imgs | rev| cut -d '/' -f 2| rev | sort -u > ../data/annotated_donor_ids
cat ../data/annotated_donor_ids | while read line
#cat ../data/annotated_donors | while read line
#cat rem | while read line
do
    #if [ "$line" = "07b" ];
    #then
    #    echo ""
    #else
    echo $line
    name="../data/sequences/"$line"_imgs"
    name2="../data/sequences/"$line"_pcaed_sequenced"
    echo $name2
    #if [ "$line" == "d85" ];
    #then
    #    continue
    #fi
    if [ -f "$name2" ];
    then
        echo $name2" already exist"
        continue
    else
        echo "staterd doing: "$line
        grep "sara_img/"$line /home/mousavi/new_naming_flat_list_img_paths_NoPS > $name
        python3 decom_sequence_generator_keras_pcaed.py --path $name --donor_id $line 
    fi
done
