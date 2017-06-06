 /* File : example.c */

#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float* get_all_radar_pts(float *output, int len, float *coords_rad_pts, int crp_x, int crp_y, float *radar_heights, int rh_x, float *model_data, int md_x, int md_y, int md_z, float *model_heights, int mh_x, int mh_y, int mh_z, float *llc_cosmo, int llc_cosmo_x, float *res_cosmo, int res_cosmo_x);
int binary_search(float *arr, int dim, float key);
float trilinear_interp(float *c_n, int c_n_x, float *v_n, int v_n_x, float *pos_radar, int p_r_x, float radar_bin_height);

float* get_all_radar_pts(float *output, int len, float *coords_rad_pts, int crp_x, int crp_y, float *radar_heights, int rh_x, float *model_data, int md_x, int md_y, int md_z, float *model_heights, int mh_x, int mh_y, int mh_z, float *llc_cosmo, int llc_cosmo_x, float *res_cosmo, int res_cosmo_x){
    
 float radar_pts[crp_x];
 int i=0;
 int llc_i=0;
 int llc_j=0;
 int current_llc_i=0;
 int current_llc_j=0;
 int flag=1;
 
 float interp_topo=0.0;
 float diff_i=0;
 float diff_j=0;
 float v[4];
 float pos_i=0.0;
 float pos_j=0.0;
 
 float pos_radar_bin[2];
 float coords_neighbours[8];
 float val_neighbours[8];

 
 int j=0;
 
 float slice_vert[mh_x];
 int k=0;
 
 int idx=0;
 int closest_1, closest_2;
 
 for(i=0;i<crp_x;i++){
    pos_radar_bin[0]=(coords_rad_pts[i*crp_y+0]-llc_cosmo[0])/res_cosmo[0];/* lon */
    pos_radar_bin[1]=(coords_rad_pts[i*crp_y+1]-llc_cosmo[1])/res_cosmo[1];/* lat */
    llc_i = floor(pos_radar_bin[1]); /* here i is the ver. coordinate (corresponding to lat) */
    llc_j = floor(pos_radar_bin[0]);  /* here j is the hor coordinate (corresponding to lon) */
    current_llc_i=llc_i;
    current_llc_j=llc_j;
    int neighb[4][2]={{llc_i,llc_j},{llc_i,llc_j+1},{llc_i+1,llc_j},{llc_i+1,llc_j+1}};
    /* get values at neighbours */
    flag=1;
    
    /* Interpolate the topography bilinearly to the radar point position */
    pos_i = fmod(pos_radar_bin[1],1.0);
    pos_j = fmod(pos_radar_bin[0],1.0);
    diff_i=1.0-pos_i;
    diff_j=1.0-pos_j;   
    for(k=0;k<4;k++){
	v[k]=model_heights[(mh_x-1)*mh_y*mh_z+neighb[k][0]*mh_z+neighb[k][1]];
    }
    interp_topo=diff_i*diff_j*v[0]+pos_i*v[2]*diff_j+diff_i*v[1]*pos_j+pos_i*pos_j*v[3];
    if(interp_topo<radar_heights[i]){
    for(j=0;j<4;j++){
	int n[2]={neighb[j][0],neighb[j][1]};
	slice_vert[mh_x];
	for(k=0;k<mh_x;k++){
		slice_vert[k]=model_heights[k*mh_y*mh_z+n[0]*mh_z+n[1]];
	    }
	idx=binary_search(slice_vert, mh_x,radar_heights[i]);
	if(idx==-1){ /* In this case one of the points is above COSMO domain*/
	    flag=2;
	    break;
	}
	if(idx==-2){ /* One of the neighbour points is below topo, so we extrapolate to the height of the radar point */
	    closest_1=mh_x-3;
	    closest_2=mh_x-2;
	}
	else{
	if(idx==(mh_x-2)){idx--;}
	closest_1=idx;
	closest_2=idx+1;
	}
	/*  printf("%d %d \n", n[0],n[1]);*/
	coords_neighbours[2*j]=slice_vert[closest_1];
	coords_neighbours[2*j+1]=slice_vert[closest_2];
	val_neighbours[2*j]=model_data[closest_1*md_z*md_y+n[0]*md_z+n[1]];
	/* printf("%f %f \n",coords_neighbours[2*j], val_neighbours[2*j]); */
	val_neighbours[2*j+1]=model_data[closest_2*md_z*md_y+n[0]*md_z+n[1]];

	}

    if(flag==2){ /* We hit the top of COSMO domain */
	for(k=i;k<crp_x;k++){output[k]=-9999;}
    }
    else{ /* Everything is fine */
	output[i]=trilinear_interp(coords_neighbours,8,val_neighbours,8,pos_radar_bin,2, radar_heights[i]);
	/* printf("output  = %f \n",output[i]);*/
    }
    }
    else{  /* We hit a mountain, all neighbour points are below topo */
	for(k=i;k<crp_x;k++){output[k]=0.0/0.0;}
    }
 }
return output;}



int binary_search(float *arr, int dim, float key){
   
   int high = 0;
   int low = dim - 1;
   int middle = 0;
   int pos_left=0;
   int pos_right=0;
    
   if(key>arr[0]){ /* We hit the top of COSMO domain */
	return -1;
   }
   else if(key<arr[dim-1]){  /* We hit a mountain */
	return -2;
   }
   while (high < low) {
      middle = (high + low)/2;   
      if (arr[middle] == key){
      return middle;
    }
      else if (low == middle || high == middle) {
         return high;
      }
      else if (arr[middle] < key)
      {low=middle;}
      else if (arr[middle] > key){
	  high=middle;}
   }
}


float trilinear_interp(float *c_n, int c_n_x, float *v_n, int v_n_x, float *pos_radar, int p_r_x, float radar_bin_height) {
    float **data;
    int i;

    float v[4];
    float interp=0;
    float diff_i=0;
    float diff_j=0;
    float pos_i=0.0;
    float pos_j=0.0;
    int j=0;
    
    for(j=0;j<c_n_x/2;j++){
	v[j]=v_n[2*j+1]-(v_n[2*j+1]-v_n[2*j])/(c_n[2*j]-c_n[2*j+1])*(radar_bin_height-c_n[2*j+1]);
	/*printf("%f %f %f %f \n",v_n[2*j+1],v_n[2*j],c_n[2*j],c_n[2*j+1]);*/
    }

    pos_i = fmod(pos_radar[1],1.0);
    pos_j = fmod(pos_radar[0],1.0);

    diff_i=1.0-pos_i;
    diff_j=1.0-pos_j;

    /* printf("pos: %f .. %f\n",x,y);*/
    interp=diff_i*diff_j*v[0]+pos_i*v[2]*diff_j+diff_i*v[1]*pos_j+pos_i*pos_j*v[3];
    return interp;
}

