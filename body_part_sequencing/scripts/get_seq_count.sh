#files=*_pcaed_sequenced
ls *_pcaed_sequenced | while read name
do
    donor_id=$(echo $name| cut -d "_" -f 1)
    echo "number_of_images_in_each_cluster,cluster_name" > /home/noah22/sequences/donor_clusters/$donor_id"_new"

    cat $donor_id"_pcaed_sequenced"| sort -t: -k2 |cut -d ":" -f 2|uniq -c >> /home/noah22/sequences/donor_clusters/$donor_id"_new"
done
