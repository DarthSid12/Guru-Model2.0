find data/thatcher_data/faces/cnn \
    -mindepth 3 -maxdepth 3 -type d | while read dir; do

    count=$(find "$dir" -maxdepth 1 -name "*.png" | wc -l)

    if [ "$count" -lt 4 ]; then
        echo "Removing $dir ($count images)"
        rm -rf "$dir"
    fi

done