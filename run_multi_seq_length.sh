for i in $( seq 2 10);
do
    sed -i "s/\"name\": .*,/\"name\": \"seq$i\",/g" configs/config.json
    sed -i "s/\"experim_name\": .*,/\"experim_name\": \"seq$i\",/g" configs/config.json
    bash body_part_sequencing/scripts/getSeqsWithLenX.sh $i
    bash train.sh
    cp /data/sara/TCT/CCT/body_part_sequencing/data/unsup_train_seq.txt saved_body_part/seq$i/
    cp /data/sara/TCT/CCT/body_part_sequencing/data/sup_train.txt saved_body_part/seq$i/
    cp /data/sara/TCT/CCT/body_part_sequencing/data/val.txt saved_body_part/seq$i/
done
