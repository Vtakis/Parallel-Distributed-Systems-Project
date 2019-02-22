
//////////////////////////////////////////////////////////
Compile with 2 ways: simple compilation or with using mpiP
//////////////////////////////////////////////////////////

	mpicc -o exe main.c conv.c

	/usr/local/mpich3/bin/mpicc -o exe main.c conv.c -L/usr/local/mpip3/lib -lmpiP -lm -lbfd -liberty

//////////////////////////////////////////////////////////////
Running program as follow: 

		mpiexec -f machines -n <n> ./exe -i <img_file> -w <width> -h <height> -l <numIter> -t <image_type>;

//////////////////////////////////////////////////////////////
e.x.

	mpiexec -f machines -n 16 ./exe -i waterfall_grey_1920_2520.raw -w 1920 -h 2520 -l 30 -t grey

	mpiexec -f machines -n 16 ./exe -i waterfall_1920_2520.raw -w 1920 -h 2520 -l 30 -t rgb

///////////////////////////////////////////////////////////////



