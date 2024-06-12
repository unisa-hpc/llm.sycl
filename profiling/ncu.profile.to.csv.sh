ncu --set basic -o profile.basic.csv ./LLM_SYCL -s --batch 1 -x -g 10 -y
ncu --set basic -o profile.basic.gold.csv ./OrigTrain -b 1
echo "Now, use the UI to save it as a .csv file, named prof.csv ."
