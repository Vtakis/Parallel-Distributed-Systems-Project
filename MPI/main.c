#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

#include "conv.h"

MPI_Comm comm;

int main(int argc, char **argv){

	int flag_i,flag_w,flag_h,flag_t,flag_l;
	int procRank,procsNum,imgWidth,imgHeight,numIter,imgSize;
	char *imgType,*img,*filterImg;
	int i,j,rowsProc,colsProc;
	int findNorth,findWest,findEast,findSouth;
	int rowBlockSplit,colBlockSplit;
	int startRow,startCol;
	uint8_t *ImgArray,*newImgArray,*tempArray;
	double timeWait,procRunTime;
	
	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &procsNum); /*Number of processes*/
  	MPI_Comm_rank(comm, &procRank); /*My process rank*/

	MPI_Status status;

	MPI_Datatype rowGREY;
	MPI_Datatype colGREY;
	MPI_Datatype rowRGB;
	MPI_Datatype colRGB;

	MPI_Request NorthSendReq;
	MPI_Request WestSendReq;
	MPI_Request EastSendReq;
	MPI_Request SouthSendReq;

	MPI_Request NorthRecvReq;
	MPI_Request WestRecvReq;
	MPI_Request EastRecvReq;
	MPI_Request SouthRecvReq;

	if(procRank==0){
		if(argc<11){
			printf("\nError! Wrong number of arguments was given!\n\n");
			printf("Please run program as follow:\n\n\tmpiexec -f machines -n <n> ./exe -i <img_file> -w <width> -h <height> -t <image_type> -l <Iter>\n\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			return EXIT_FAILURE;
		}
	}

	flag_i=-1;
	flag_w=-1;
	flag_h=-1;
	flag_t=-1;
	flag_l=-1;

	for(i=0;i<argc;i++){
		if(!strcmp(argv[i],"-i"))
			flag_i=i+1;
		if(!strcmp(argv[i],"-w"))
			flag_w=i+1;
		if(!strcmp(argv[i],"-h"))
			flag_h=i+1;
		if(!strcmp(argv[i],"-t"))
			flag_t=i+1;
		if(!strcmp(argv[i],"-l"))
			flag_l=i+1;
	}

	if(procRank==0){	
		if(flag_i==-1){
			printf("\nERROR! Please give image file!\n\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
		if(flag_w==-1){
			printf("\nERROR! Please give image's width!\n\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
		if(flag_h==-1){
			printf("\nERROR! Please give image's height!\n\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
		if(flag_t==-1){
			printf("\nERROR! Please give image's type (grey or rgb)!\n\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
		if(flag_l==-1){
			printf("\nERROR! Please give number of iterations of convolution!\n\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
	}
///////////////////////////////////////////////////////////////////////////
	imgSize=strlen(argv[flag_i]+1);
	img=malloc(imgSize);
	if(img==NULL){
		printf("ERROR! Out of memory at %d process!\n",procRank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
       	return EXIT_FAILURE;
	}
	strcpy(img,argv[flag_i]);
	imgWidth=atoi(argv[flag_w]);
	imgHeight=atoi(argv[flag_h]);
	numIter=atoi(argv[flag_l]);
	if(!strcmp(argv[flag_t],"grey")||!strcmp(argv[flag_t],"GREY")){
		imgType=malloc(5);
		if(imgType==NULL){
			printf("ERROR! Out of memory at %d process!\n",procRank);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	    	return EXIT_FAILURE;
		}
		strcpy(imgType,"GREY");
	}
	else if(!strcmp(argv[flag_t],"rgb")||!strcmp(argv[flag_t],"RGB")){
		imgType=malloc(5);
		if(imgType==NULL){
			printf("ERROR! Out of memory at %d process!\n",procRank);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	    	return EXIT_FAILURE;
		}
		strcpy(imgType,"RGB");
	}
	else{
		printf("ERROR! Please give 'grey' or 'rgb' as image type\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}


	if(procRank==0){
		rowBlockSplit=-1;
		colBlockSplit=-1;
		howToSplitImage(imgHeight,imgWidth,procsNum,&rowBlockSplit,&colBlockSplit);

		if(rowBlockSplit==-1||colBlockSplit==-1){
			printf("ERROR! Can't equally split image to given number of processes!\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			return EXIT_FAILURE;
		}
		printf("\nrowBlockSplit = %d - colBlockSplit = %d \n",rowBlockSplit,colBlockSplit);

	}

	MPI_Bcast(&imgWidth,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&imgHeight,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&numIter,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(imgType,6,MPI_CHAR,0,MPI_COMM_WORLD);
	MPI_Bcast(&rowBlockSplit,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&colBlockSplit,1,MPI_INT,0,MPI_COMM_WORLD);

	rowsProc = imgHeight/rowBlockSplit;//rows per process
	colsProc = imgWidth/colBlockSplit;//columns per process

	MPI_Type_vector(rowsProc,1,colsProc+2,MPI_BYTE,&colGREY);
	MPI_Type_commit(&colGREY);
	MPI_Type_contiguous(colsProc,MPI_BYTE,&rowGREY);
	MPI_Type_commit(&rowGREY);

	MPI_Type_vector(rowsProc,3,3*(colsProc+2),MPI_BYTE,&colRGB);
	MPI_Type_commit(&colRGB);
	MPI_Type_contiguous(3*colsProc,MPI_BYTE,&rowRGB);
	MPI_Type_commit(&rowRGB);

	float **filter = malloc(3*sizeof(float*));
	for(i=0;i<3;i++)
		filter[i]=malloc(3*sizeof(float));

	int defaultFilter[3][3]={{1,2,1},{2,4,2},{1,2,1}};;
	
	/*srand(time(NULL));
	//use this for random filter 3x3
	for(i=0;i<3;i++){
		for(j=0;j<3;j++)
			defaultFilter[i][j] = rand()%10+1;
	}
	*/
	int sumFilter=0;

	for(i=0;i<3;i++){
		for(j=0;j<3;j++)
			sumFilter+=defaultFilter[i][j];
	}

	for(i=0;i<3;i++){
		for(j=0;j<3;j++){
			filter[i][j]=defaultFilter[i][j]/(float)sumFilter;
		}
	}
	
	ImgArray=NULL;
	newImgArray=NULL;

	MPI_File fileMPI;

	if(!strcmp(imgType,"GREY")){
		ImgArray=malloc((rowsProc+2)*(colsProc+2)*sizeof(uint8_t));
		newImgArray=malloc((rowsProc+2)*(colsProc+2)*sizeof(uint8_t));

		if(ImgArray==NULL||newImgArray==NULL){
			printf("ERROR! Out of memory at %d process!\n",procRank);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
	}
	else{//RGB
		ImgArray=malloc((rowsProc+2)*3*(colsProc+2)*sizeof(uint8_t));
		newImgArray=malloc((rowsProc+2)*3*(colsProc+2)*sizeof(uint8_t));

		if(ImgArray==NULL||newImgArray==NULL){
			printf("ERROR! Out of memory at %d process!\n",procRank);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		   	return EXIT_FAILURE;
		}
	}

	startRow = (procRank/colBlockSplit)*rowsProc;
	startCol = (procRank%colBlockSplit)*colsProc;

	//open image file
	MPI_File_open(MPI_COMM_WORLD, img, MPI_MODE_RDONLY, MPI_INFO_NULL, &fileMPI);
	if(fileMPI==NULL){
		printf("ERROR! Cannot open file at %d process!\n",procRank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}

	if(!strcmp(imgType,"GREY")){
		for(i=1;i<=rowsProc;i++){
			MPI_File_seek(fileMPI,(startRow+i-1)*imgWidth+startCol,MPI_SEEK_SET);
			MPI_File_read(fileMPI,getPixel(ImgArray,i,1,colsProc+2),colsProc,MPI_BYTE,&status);
		}
	}
	else{//RGB
		for(i=1;i<=rowsProc;i++){
			MPI_File_seek(fileMPI,3*((startRow+i-1)*imgWidth+startCol),MPI_SEEK_SET);
			MPI_File_read(fileMPI,getPixel(ImgArray,i,3,3*(colsProc+2)),3*colsProc,MPI_BYTE,&status);
		}
	}
	MPI_File_close(&fileMPI);

	findNorth = -1;
	findWest = -1;
	findEast = -1;
	findSouth = -1;

	if(startRow!=0)
		findNorth = procRank-colBlockSplit;
	if(startCol!=0)
		findWest = procRank-1;
	if(startCol+colsProc<imgWidth)
		findEast = procRank+1;
	if(startRow+rowsProc<imgHeight)
		findSouth = procRank+colBlockSplit;

	MPI_Barrier(MPI_COMM_WORLD);
	//printf("%s proc %d\n",imgType,procRank);

	timeWait = MPI_Wtime();

	for(i=0;i<numIter;i++){//run convolution for 'numIter' times
		if(!strcmp(imgType,"GREY")){
			if(findNorth!=-1){
				MPI_Isend(getPixel(ImgArray,1,1,colsProc+2),1,rowGREY,findNorth,0,MPI_COMM_WORLD,&NorthSendReq);
				MPI_Irecv(getPixel(ImgArray,0,1,colsProc+2),1,rowGREY,findNorth,0,MPI_COMM_WORLD,&NorthRecvReq);
			}
			if(findWest!=-1){
				MPI_Isend(getPixel(ImgArray,1,1,colsProc+2),1,colGREY,findWest,0,MPI_COMM_WORLD,&WestSendReq);
				MPI_Irecv(getPixel(ImgArray,1,0,colsProc+2),1,colGREY,findWest,0,MPI_COMM_WORLD,&WestRecvReq);
			}
			if(findEast!=-1){
				MPI_Isend(getPixel(ImgArray,1,colsProc,colsProc+2),1,colGREY,findEast,0,MPI_COMM_WORLD,&EastSendReq);
				MPI_Irecv(getPixel(ImgArray,1,colsProc+1,colsProc+2),1,colGREY,findEast,0,MPI_COMM_WORLD,&EastRecvReq);
			}
			if(findSouth!=-1){
				MPI_Isend(getPixel(ImgArray,rowsProc,1,colsProc+2),1,rowGREY,findSouth,0,MPI_COMM_WORLD,&SouthSendReq);
				MPI_Irecv(getPixel(ImgArray,rowsProc+1,1,colsProc+2),1,rowGREY,findSouth,0,MPI_COMM_WORLD,&SouthRecvReq);
			}
		}
		else{//RGB
			if(findNorth!=-1){
				MPI_Isend(getPixel(ImgArray,1,3,3*(colsProc+2)),1,rowRGB,findNorth,0,MPI_COMM_WORLD,&NorthSendReq);
				MPI_Irecv(getPixel(ImgArray,0,3,3*(colsProc+2)),1,rowRGB,findNorth,0,MPI_COMM_WORLD,&NorthRecvReq);
			}
			if(findWest!=-1){
				MPI_Isend(getPixel(ImgArray,1,3,3*(colsProc+2)),1,colRGB,findWest,0,MPI_COMM_WORLD,&WestSendReq);
				MPI_Irecv(getPixel(ImgArray,1,0,3*(colsProc+2)),1,colRGB,findWest,0,MPI_COMM_WORLD,&WestRecvReq);
			}
			if(findEast!=-1){
				MPI_Isend(getPixel(ImgArray,1,3*colsProc,3*(colsProc+2)),1,colRGB,findEast,0,MPI_COMM_WORLD,&EastSendReq);
				MPI_Irecv(getPixel(ImgArray,1,3*(colsProc+1),3*(colsProc+2)),1,colRGB,findEast,0,MPI_COMM_WORLD,&EastRecvReq);
			}
			if(findSouth!=-1){
				MPI_Isend(getPixel(ImgArray,rowsProc,3,3*(colsProc+2)),1,rowRGB,findSouth,0,MPI_COMM_WORLD,&SouthSendReq);
				MPI_Irecv(getPixel(ImgArray,rowsProc+1,3,3*(colsProc+2)),1,rowRGB,findSouth,0,MPI_COMM_WORLD,&SouthRecvReq);
			}
		}

		//convolution for block of current process
		convolution(ImgArray,newImgArray,1,rowsProc,1,colsProc,colsProc,filter,imgType);

		//wait to receive request and then make convolution
		if(findNorth!=-1){
			MPI_Wait(&NorthRecvReq,&status);
			convolution(ImgArray,newImgArray,1,1,2,colsProc-1,colsProc,filter,imgType);
		}
		if(findWest!=-1){
			MPI_Wait(&WestRecvReq,&status);
			convolution(ImgArray,newImgArray,2,rowsProc-1,1,1,colsProc,filter,imgType);
		}
		if(findEast!=-1){
			MPI_Wait(&EastRecvReq,&status);
			convolution(ImgArray,newImgArray,2,rowsProc-1,colsProc,colsProc,colsProc,filter,imgType);
		}
		if(findSouth!=-1){
			MPI_Wait(&SouthRecvReq,&status);
			convolution(ImgArray,newImgArray,rowsProc,rowsProc,2,colsProc-1,colsProc,filter,imgType);
		}

		//convolute on corners (NW-NE-SW-SE)
		if(findNorth!=-1 && findWest!=-1)
			convolution(ImgArray,newImgArray,1,1,1,1,colsProc,filter,imgType);
		if(findNorth!=-1 && findEast!=-1)
			convolution(ImgArray,newImgArray,1,1,colsProc,colsProc,colsProc,filter,imgType);
		if(findWest!=-1 && findSouth!=-1)
			convolution(ImgArray,newImgArray,rowsProc,rowsProc,1,1,colsProc,filter,imgType);
		if(findEast!=-1 && findSouth!=-1)
			convolution(ImgArray,newImgArray,rowsProc,rowsProc,colsProc,colsProc,colsProc,filter,imgType);

		//wait to send requests
		if(findNorth!=-1)
			MPI_Wait(&NorthSendReq,&status);
		if(findWest!=-1)
			MPI_Wait(&WestSendReq,&status);
		if(findEast!=-1)
			MPI_Wait(&EastSendReq,&status);
		if(findSouth!=-1)
			MPI_Wait(&SouthSendReq,&status);

		//swapping current image array with new image array
		tempArray=NULL;
		tempArray=ImgArray;
		ImgArray=newImgArray;
		newImgArray=tempArray;		

	}
	timeWait = MPI_Wtime()-timeWait;
	//printf("timeWait %f for process %d\n",timeWait,procRank);

	//create filtered image
	filterImg = malloc(imgSize+8);
	sprintf(filterImg,"filter_%s",img);
	MPI_File filterFileMPI;
	MPI_File_open(MPI_COMM_WORLD, filterImg, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &filterFileMPI);
	if(filterFileMPI==NULL){
		printf("ERROR! Cannot open file at %d process!\n",procRank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}
	if (!strcmp(imgType,"GREY")) {
		for(i=1;i<=rowsProc;i++){
			MPI_File_seek(filterFileMPI,(startRow+i-1)*imgWidth+startCol,MPI_SEEK_SET);
			MPI_File_write(filterFileMPI,getPixel(ImgArray,i,1,colsProc+2),colsProc,MPI_BYTE,MPI_STATUS_IGNORE);
		}
	}
	else{
		for (i=1;i<=rowsProc;i++) {
			MPI_File_seek(filterFileMPI,3*((startRow+i-1)*imgWidth+startCol),MPI_SEEK_SET);
			MPI_File_write(filterFileMPI,getPixel(ImgArray,i,3,3*(colsProc+2)),3*colsProc,MPI_BYTE,MPI_STATUS_IGNORE);
		}
	}
	MPI_File_close(&filterFileMPI);

	if(procRank!=0)
		MPI_Send(&timeWait,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
	else{
		for(i=1;i!=procsNum;++i){
			MPI_Recv(&procRunTime,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&status);
        	//printf("process %d: %f\n",i, procRunTime);
			if(procRunTime>timeWait)
				timeWait=procRunTime;
		}
        printf("%f\n",timeWait);
	}

//////////////////
	for (i=0;i<3;i++)
		free(filter[i]);
	free(filter);
	free(img);
	free(ImgArray);
	free(newImgArray);
	free(filterImg);
	free(fileMPI);
	free(imgType);
	MPI_Type_free(&rowGREY);
	MPI_Type_free(&colGREY);
	MPI_Type_free(&rowRGB);
	MPI_Type_free(&colRGB);

	MPI_Finalize();
	return EXIT_SUCCESS;
}
