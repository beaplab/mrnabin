paste <(awk '/^[^#]/ {print $1,$2,$8,$11}' "$1") <(awk '/^[^#]/ {print $8,$11}' "$2") |
    sed 's/[()]//g' |
    awk 'BEGIN{OFS="\t"; d=0} \
    	{a+=($5-$3); b+=($6-$4); left+=$3; right+=$5; if ($5-$3 > c) c=$5-$3; if($5-$3 < d){ d=$5-$3 };  print $1,$2,$3,$4,$5,$6,$5-$3,$6-$4} \
	END{print "avg score diff:",a/NR,"\nmax score diff:",c,"\nmin score diff:",d,"\navg coverage diff:",b/NR,"\nleft avg score:",left/NR,"\nright avg score:",right/NR}'

