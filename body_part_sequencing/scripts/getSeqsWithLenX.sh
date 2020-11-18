# At the end you need to edit the file and add /usb/ to each image_name
seq_len=$1
grep -w $seq_len /data/sara/TCT/CCT/body_part_sequencing/data/sequences/all_seqs_lengths | shuf | rev|cut -d " " -f 1|rev > ../data/seqsNamesWithLen$seq_len"Shuffed"
i=0
rm /data/sara/TCT/CCT/body_part_sequencing/data/unsup_train_seq.txt
cat /data/sara/TCT/CCT/body_part_sequencing/data/seqsNamesWithLen$seq_len"Shuffed" | while read line
do
	all_seq=""
	imgs=$(grep -w $line /data/sara/TCT/CCT/body_part_sequencing/data/sequences/*pcaed_sequenced | cut -d ":" -f 2| rev| cut -d "/" -f 1| rev | sed 's/.icon//' )
	echo $imgs >> /data/sara/TCT/CCT/body_part_sequencing/data/unsup_train_seq.txt
	####### to copy these images on /usb
	#grep -w $line ../data/sequences/*pcaed_sequenced |cut -d ":" -f 2 | while read imgs
	#do
	#	echo $imgs >> ../data/dataUSBseqsWithLen$seq_len
	#done
	((i+=1))
	if [[ i -gt 2559 ]]
	then
		break
	fi
done


sed -i 's/^/\/usb\//g' /data/sara/TCT/CCT/body_part_sequencing/data/unsup_train_seq.txt
sed -i 's/ / \/usb\//g' /data/sara/TCT/CCT/body_part_sequencing/data/unsup_train_seq.txt
