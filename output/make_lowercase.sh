for file in *.csv
do
    cat $file | tr '[:upper:]' '[:lower:]' | sponge $file
done
