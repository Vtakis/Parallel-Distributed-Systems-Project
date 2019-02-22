#ifndef CONV_H
#define CONV_H

void howToSplitImage(int,int,int,int *,int *);
uint8_t *getPixel(uint8_t *,int,int,int);
void convolutionGREY(uint8_t *,uint8_t *,int,int,int,float **);
void convolutionRGB(uint8_t *,uint8_t *,int,int,int,float **);
void convolution(uint8_t *,uint8_t *,int,int,int,int,int,float **,char*);

#endif
