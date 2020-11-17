# At the end you need to edit the file and add /usb/ to each image_name
seq_len=5
#grep -w $seq_len /data/sara/CCT/body_part_data/data/donor_clusters_counts/* | shuf | rev|cut -d " " -f 1|rev > ../data/seqsNamesWithLen$seq_len"Shuffed"
i=0
cat ../data/seqsNamesWithLen$seq_len"Shuffed" | while read line
do
	all_seq=""
	imgs=$(grep -w $line ../data/sequences/*pcaed_sequenced | cut -d ":" -f 2| rev| cut -d "/" -f 1| rev | sed 's/.icon//' )
	#echo $imgs >> ../data/unsup_train_$seq_len"seq.txt"
	####### to copy these images on /usb
	grep -w $line ../data/sequences/*pcaed_sequenced |cut -d ":" -f 2 | while read imgs
	do
		echo $imgs >> ../data/dataUSBseqsWithLen$seq_len
	done
	((i+=1))
	if [[ i -gt 2560 ]]
	then
		break
	fi

done


#sed -i 's/^/\/usb\//g' ../data/unsup_train_$seq_len"seq.txt"
#sed -i 's/ / \/usb\//g' ../data/unsup_train_$seq_len"seq.txt"
