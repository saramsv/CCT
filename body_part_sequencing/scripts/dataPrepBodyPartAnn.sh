name=$(echo $1 | rev | cut -d'/' -f 1 | rev)

echo "_id,user,location,image,tag,created,__v" > ../data/bodyPartAnn$name

cat $1 | grep poly | grep -E "foot|leg|hand|arm|torso|head" >> ../data/bodyPartAnn$name

python3 fix_tags.py ../data/bodyPartAnn$name > ../data/bodyPartAnn$name"FixedLabels"

sed -i 's/"/""/g' ../data/bodyPartAnn$name"FixedLabels"
sed -i 's/\[{""type/"\[{""type/g' ../data/bodyPartAnn$name"FixedLabels"
sed -i 's/\],sara/\]",sara/g' ../data/bodyPartAnn$name"FixedLabels" 

python3 generate_annotated_images.py ../data/bodyPartAnn$name"FixedLabels"
