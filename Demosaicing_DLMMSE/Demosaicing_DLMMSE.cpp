#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;

void save_image(const cv::Mat& image, const std::string& filename) {
    string folderPath = "Data/";
    string imagePath = folderPath + filename + ".png";
    string csvPath = folderPath + filename + ".csv";

    // Save image
    imwrite(imagePath, image);

    // Save data to CSV (assuming image is a Mat of CV_8UC3)
    ofstream csvFile(csvPath);
    if (csvFile.is_open()) {
        csvFile << format(image, cv::Formatter::FMT_CSV) << endl;
        csvFile.close();
    }
    else {
        cerr << "Error opening CSV file" << endl;
    }
}

cv::Mat convolve1d(const cv::Mat& img, const std::vector<float>& kernel) {
    cv::Mat result;
    cv::Mat kernelMat = cv::Mat(kernel).reshape(1, 1);
    cv::filter2D(img, result, -1, kernelMat, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    return result;
}

void modify_alternating_rows_and_cols(cv::Mat& mat, bool is_row) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            if (is_row) {
                if ((i % 2 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 2 == 0)) {
                    mat.at<float>(i, j) *= -1;
                }
            }
            else {
                if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {
                    mat.at<float>(i, j) *= -1;
                }
            }
        }
    }
}

cv::Mat Demosaic_DLMMSE(const cv::Mat& img) {
    cv::Mat new_img;
    img.convertTo(new_img, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(new_img, channels);
    cv::Mat R = channels[0], G = channels[1], B = channels[2];
    cv::Mat S = R + G + B;

    // Step 1: Interpolate G
    // Step 1.1: Interpolate by a simple method, then compute G-R and G-B (2-2 ~ 2-6 in paper)
    std::vector<float> kernel1 = { -1, 2, 2, 2, -1 };
    cv::Mat H = convolve1d(S, kernel1) / 4;
    cv::Mat V = convolve1d(S.t(), kernel1).t() / 4;

    cv::Mat delta_H = H - S;
    modify_alternating_rows_and_cols(delta_H, true);
    cv::Mat delta_V = V - S;
    modify_alternating_rows_and_cols(delta_V, true);

    // Step 1.2: Use a low-pass filter (3-4)
    std::vector<float> gaussian_filter = { 4, 9, 15, 23, 26, 23, 15, 9, 4 };
    cv::Mat gaussian_H = convolve1d(delta_H, gaussian_filter) / cv::sum(gaussian_filter)[0];
    cv::Mat gaussian_V = convolve1d(delta_V.t(), gaussian_filter).t() / cv::sum(gaussian_filter)[0];

    // Step 1.3: Calculate mean_x, var_x, var_v (2-12, 3-6 ~ 3-10)
    std::vector<float> mean_filter(9, 1);
    cv::Mat mean_H = convolve1d(gaussian_H, mean_filter) / cv::sum(mean_filter)[0];
    cv::Mat mean_V = convolve1d(gaussian_V.t(), mean_filter).t() / cv::sum(mean_filter)[0];

    // Ensure that the dimensions match by resizing the convolution result to match the original matrix dimensions
    cv::Mat var_value_H = convolve1d((gaussian_H - mean_H).mul(gaussian_H - mean_H), mean_filter);
    cv::resize(var_value_H, var_value_H, gaussian_H.size(), 0, 0, cv::INTER_LINEAR);
    var_value_H = var_value_H / cv::sum(mean_filter)[0] + 1e-10;

    // Vertical processing part
    cv::Mat transposed_gaussian_V = gaussian_V.t();
    cv::Mat transposed_mean_V = mean_V.t();
    cv::Mat transposed_delta_V = delta_V.t();

    // Compute var_value_V
    cv::Mat var_value_V_temp = convolve1d((transposed_gaussian_V - transposed_mean_V).mul(transposed_gaussian_V - transposed_mean_V), mean_filter);
    cv::resize(var_value_V_temp, var_value_V_temp, transposed_gaussian_V.size(), 0, 0, cv::INTER_LINEAR);
    var_value_V_temp = var_value_V_temp / cv::sum(mean_filter)[0] + 1e-10;
    cv::Mat var_value_V = var_value_V_temp.t();

    cv::Mat var_noise_H = convolve1d((gaussian_H - delta_H).mul(gaussian_H - delta_H), mean_filter);
    cv::resize(var_noise_H, var_noise_H, gaussian_H.size(), 0, 0, cv::INTER_LINEAR);
    var_noise_H = var_noise_H / cv::sum(mean_filter)[0] + 1e-10;

    // Compute var_noise_V
    cv::Mat var_noise_V_temp = convolve1d((transposed_gaussian_V - transposed_delta_V).mul(transposed_gaussian_V - transposed_delta_V), mean_filter);
    cv::resize(var_noise_V_temp, var_noise_V_temp, transposed_gaussian_V.size(), 0, 0, cv::INTER_LINEAR);
    var_noise_V_temp = var_noise_V_temp / cv::sum(mean_filter)[0] + 1e-10;
    cv::Mat var_noise_V = var_noise_V_temp.t();

    // Step 1.4: make delta more precise by 2-12 in paper
    // Compute new_H and new_V with element-wise operations
    cv::Mat new_H = mean_H + (var_value_H / (var_noise_H + var_value_H)).mul(delta_H - mean_H);
    cv::Mat new_V = mean_V + (var_value_V / (var_noise_V + var_value_V)).mul(delta_V - mean_V);

    // Step 1.5: combine delta of two directions to make more precise by 3-11 and 4-7 in paper
    cv::Mat var_x_H = cv::abs(var_value_H - var_value_H / (var_value_H + var_noise_H)) + 1e-10;
    cv::Mat var_x_V = cv::abs(var_value_V - var_value_V / (var_value_V + var_noise_V)) + 1e-10;
    cv::Mat w_H = var_x_V / (var_x_H + var_x_V);
    cv::Mat w_V = var_x_H / (var_x_H + var_x_V);
    cv::Mat final_result = w_H.mul(new_H) + w_V.mul(new_V);

    // Step 1.6: add delta, ok~
    for (int i = 0; i < new_img.rows; i += 2) {
        for (int j = 0; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[1] = R.at<float>(i, j) + final_result.at<float>(i, j);
        }
    }
    for (int i = 1; i < new_img.rows; i += 2) {
        for (int j = 1; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[1] = B.at<float>(i, j) + final_result.at<float>(i, j);
        }
    }

    // Split the new_img into its three channels
    cv::split(new_img, channels);
    R = channels[0];
    G = channels[1];
    B = channels[2];

    // Calculate G - R and G - B
    cv::Mat G_minus_R = G - R;
    cv::Mat G_minus_B = G - B;

    // Step 2.1: R in B or B in R (Figure.6)
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 0, 1, 0, 0, 0, 1, 0, 1);
    cv::Mat delta_GR, delta_GB;
    cv::filter2D(G_minus_R, delta_GR, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(G_minus_B, delta_GB, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    delta_GR /= 4;
    delta_GB /= 4;
    for (int i = 1; i < new_img.rows; i += 2) {
        for (int j = 1; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[0] = G.at<float>(i, j) - delta_GR.at<float>(i, j);
        }
    }
    for (int i = 0; i < new_img.rows; i += 2) {
        for (int j = 0; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[2] = G.at<float>(i, j) - delta_GB.at<float>(i, j);
        }
    }

    // Step 2.2: R/B in G
    cv::split(new_img, channels);
    R = channels[0];
    B = channels[2];
    kernel = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0);
    // Ensure G, R, and B are of type CV_32F

    G_minus_R = G - R;
    G_minus_B = G - B;

    cv::filter2D(G_minus_R, delta_GR, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(G_minus_B, delta_GB, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    delta_GR /= 4;
    delta_GB /= 4;
    for (int i = 0; i < new_img.rows; i += 2) {
        for (int j = 1; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[0] = G.at<float>(i, j) - delta_GR.at<float>(i, j);
        }
    }
    for (int i = 1; i < new_img.rows; i += 2) {
        for (int j = 0; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[0] = G.at<float>(i, j) - delta_GR.at<float>(i, j);
        }
    }
    for (int i = 0; i < new_img.rows; i += 2) {
        for (int j = 1; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[2] = G.at<float>(i, j) - delta_GB.at<float>(i, j);
        }
    }
    for (int i = 1; i < new_img.rows; i += 2) {
        for (int j = 0; j < new_img.cols; j += 2) {
            new_img.at<cv::Vec3f>(i, j)[2] = G.at<float>(i, j) - delta_GB.at<float>(i, j);
        }
    }

    cv::Mat final_img;
    new_img.convertTo(final_img, CV_8U, 1.0, 0.5);
    return final_img;
}

Mat make_bayer(const Mat& img) {
    Mat new_img = Mat::zeros(img.size(), img.type());

    // Accessing pixel values directly for efficiency
    for (int y = 0; y < img.rows; y += 2) {
        for (int x = 0; x < img.cols; x += 2) {
            new_img.at<Vec3b>(y, x)[0] = img.at<Vec3b>(y, x)[0]; // R
            new_img.at<Vec3b>(y, x + 1)[1] = img.at<Vec3b>(y, x + 1)[1]; // G
            new_img.at<Vec3b>(y + 1, x)[1] = img.at<Vec3b>(y + 1, x)[1]; // G
            new_img.at<Vec3b>(y + 1, x + 1)[2] = img.at<Vec3b>(y + 1, x + 1)[2]; // B
        }
    }

    return new_img;
}

int main() {
    Mat img = imread("Camera_Test.png", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    cvtColor(img, img, COLOR_BGR2RGB);

    Mat Bayer = make_bayer(img);

    save_image(Bayer, "Bayer");

    Mat Demosaiced = Demosaic_DLMMSE(Bayer);

    cvtColor(Demosaiced, Demosaiced, COLOR_RGB2BGR);

    save_image(Demosaiced, "DLMMSE");

    imshow("DLMMSE Image", Demosaiced);
    waitKey(0);

    return 0;
}
