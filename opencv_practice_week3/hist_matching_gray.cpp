//
//  hist_matching_gray.cpp
//  opencv_practice_week3
//
//  Created by 류예나 on 2018. 3. 19..
//  Copyright © 2018년 류예나. All rights reserved.
//

#include "hist_func.h"


void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);//HE function

int main() {
    
    Mat target = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week3/opencv_practice_week3/opencv_practice_week3/homework_image.jpg", CV_LOAD_IMAGE_COLOR);
    Mat reference = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week3/opencv_practice_week3/opencv_practice_week3/homework_image_streching_gray.jpg");
    
    cvtColor(target, target, CV_RGB2GRAY);
    
    Mat equalized_tgt = target.clone();
    Mat equalized_ref = reference.clone();
    
    float *PDF_tgt = cal_PDF(target);
    float *PDF_ref = cal_PDF(reference);
    float *CDF_tgt = cal_CDF(target);
    float *CDF_ref = cal_CDF(reference);
    
    G trans_func_eq_tgt[L] = {0};
    G trans_func_eq_ref[L] = {0};
    
    hist_eq(target, equalized_tgt, trans_func_eq_tgt, CDF_tgt);
    hist_eq(reference, equalized_ref, trans_func_eq_ref, CDF_ref);
    
    float *equalized_PDF_tgt = cal_PDF(equalized_tgt);
    float *equalized_PDF_ref = cal_PDF(equalized_ref);
    
//    for (int i = 0; i < L; i++){
//        fprintf(f_PDF_target, "%d\t%f\n", i, PDF_tgt[i]);
//        fprintf(f_equalized_PDF_tgt, "%d\t%f\n", i, equalized_PDF_tgt[i]);
//        fprintf(f_equalized_PDF_ref, "%d\t%f\n", i, equalized_PDF_ref[i]);
//
//        fprintf(f_trans_func_eq_target, "%d\t%f\n", i, trans_func_eq_tgt[i]);
//    }
    
    free(PDF_tgt);
    free(PDF_ref);
    free(CDF_tgt);
    free(CDF_ref);
    
    namedWindow("target", WINDOW_AUTOSIZE);
    imshow("target", equalized_tgt);

    namedWindow("reference", WINDOW_AUTOSIZE);
    imshow("reference", reference);

    waitKey(0);
    
    return 0;
    
   
}

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF) {//HE
    
    for (int i = 0; i < L; i++)
        trans_func[i] = (G)((L - 1) * CDF[i]);

    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++){
            equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
        }

}

void hist_inv(Mat &input, Mat &matched, G *trans_func, float *CDF){
    
}
