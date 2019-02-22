#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"
#include "conv.h"

//Select best way to split image into blocks based on number of processes
void howToSplitImage(int rows,int cols,int numProc,int *useBlockRows,int *useBlockCols){
	int splitBlockRow,splitBlockCol,minPixels,countPixels;
	minPixels = rows+cols+1;
	for(splitBlockRow=1;splitBlockRow<=numProc;splitBlockRow++){
		if(numProc%splitBlockRow!=0)//cannot split image by 'splitBlockRow' and numProc processes
			continue;
		if(rows%splitBlockRow!=0)//cannot split image's rows by 'splitBlockRow' rows
			continue;
		splitBlockCol = numProc/splitBlockRow;
		if(cols%splitBlockCol!=0)//cannot split image by 'splitBlockCol' columns
			continue;
		countPixels=rows/splitBlockRow+cols/splitBlockCol;
		if(countPixels < minPixels){//select combo with minimum number of pixels on each block
			minPixels = countPixels;
			*useBlockRows = splitBlockRow;
			*useBlockCols = splitBlockCol;
		}
	}
}

uint8_t *getPixel(uint8_t *imgArray,int x,int y,int width){
	return &imgArray[width*x+y];
}

//convolution for GREY images
void convolutionGREY(uint8_t *oldImg,uint8_t *newImg,int x,int y,int imgWidth,float **filter){
	float value=0;
	int offset,i,j,p,q,off;
	offset=imgWidth*x+y;

	for(i=x-1,p=0;i<=x+1;i++,p++){
		for(j=y-1,q=0;j<=y+1;j++,q++){
			off=imgWidth*i+j;
			value+=oldImg[off]*filter[p][q];
		}
	}
	newImg[offset]=value;
}

//convolution for RGB images
void convolutionRGB(uint8_t *oldImg,uint8_t *newImg,int x,int y,int imgWidth,float **filter){
	float valueR=0,valueG=0,valueB=0;
	int offset,i,j,p,q,off;
	offset=imgWidth*x+y;

	for(i=x-1,p=0;i<=x+1;i++,p++){
		for(j=y-3,q=0;j<=y+3;j+=3,q++){
			off=imgWidth*i+j;
			valueR+=oldImg[off]*filter[p][q];
			valueG+=oldImg[off+1]*filter[p][q];
			valueB+=oldImg[off+2]*filter[p][q];			
		}
	}
	newImg[offset]=valueR;
	newImg[offset+1]=valueG;
	newImg[offset+2]=valueB;
}

void convolution(uint8_t *oldImg,uint8_t *newImg,int startRow,int endRow,int startCol,int endCol,int imgWidth,float **filter,char *imgType){
	int i,j;

	if(!strcmp(imgType,"GREY")){
#pragma omp parallel for shared(oldImg, newImg) schedule(static) collapse(2)
		for(i=startRow;i<=endRow;i++)
			for(j=startCol;j<=endCol;j++)
				convolutionGREY(oldImg,newImg,i,j,imgWidth+2,filter);
	}else{
#pragma omp parallel for shared(oldImg, newImg) schedule(static) collapse(2)
		for(i=startRow;i<=endRow;i++)
			for(j=startCol;j<=endCol;j++)
				convolutionRGB(oldImg,newImg,i,3*j,3*(imgWidth+2),filter);
	} 
	
}