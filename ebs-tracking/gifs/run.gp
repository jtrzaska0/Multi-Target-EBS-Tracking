# Set the terminal to GIF and setup the files.
set terminal gif animate delay 5
set xrange [0:346]
set yrange [0:260]
unset key
stats "eframes.txt" name "FILE"

set output "name.gif"

# Set colorbar
set palette defined (0 "red", 0.5 "red", 0.5 "green", 1 "green")
set cbtics ("0" 0.25, "1" 0.75)

# Make the gif
do for [j=0:int(FILE_blocks-2)] {plot "eframes.txt" index j using 1:2:3 ls 5 ps 0.7 palette, "tframes.txt" index j u 1:2 lt 3 lc "black" }

set output
