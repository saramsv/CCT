#!/bin/bash
# got AnnDulImgsNoSPSameDayasAnns by using only the date part of the image names on the annotated list and greping for them in the unannotated list
bash grep.sh AnnDulImgsNoSPSameDayasAnns ../../Post-mortem-Interval-Estimator/PMIs > AnnDulImgsNoSPSameDayasAnnsPMIs
bash grep.sh AnnDulImgsNoSPSameDayasAnns  new_naming_flat_list_img_paths_preds > AnnDulImgsNoSPSameDayasAnnsPred

bash grep.sh body_part_imgs ../../Post-mortem-Interval-Estimator/PMIs > body_part_imgs_pMIs
bash grep.sh body_part_imgs new_naming_flat_list_img_paths_preds > body_part_imgs_preds
cp AnnDulImgsNoSPSameDayasAnnsPMIs Ann_unAnn_samedays_pmi
cat body_part_imgs_pMIs >> Ann_unAnn_samedays_pmi

cp AnnDulImgsNoSPSameDayasAnnsPred Ann_unAnn_samedays_preds
cat body_part_imgs_preds >> Ann_unAnn_samedays_preds

join -t: <(sort -t: Ann_unAnn_samedays_preds) <(sort -t: Ann_unAnn_samedays_pmi) | sort -u > joint
