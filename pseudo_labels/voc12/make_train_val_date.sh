#bash make_train_val_date.sh all_labeled_body_part_new_naming_cleaned_noBody
num_lines=$(wc -l $1)
num=$(echo $num_lines| cut -d " " -f 1)

head -n  $(( $num*3/4 )) $1 | cut -d ":" -f 1 > train.txt
tail -n  $(( $num*1/4 )) $1 | cut -d ":" -f 1 > val.txt
