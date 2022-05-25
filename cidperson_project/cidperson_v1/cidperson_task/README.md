# train cidperson model
sh train.sh $erp $dt

# push aiflow 
sh cidperson_aiflow.sh $erp $dt
