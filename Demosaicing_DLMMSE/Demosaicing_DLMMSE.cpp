#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;

// ![get-psnr]
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
// ![get-psnr]

// ![get-mssim]

Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);        // I2^2
    Mat I1_2 = I1.mul(I1);        // I1^2
    Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}
// ![get-mssim]

struct ImageMetrics {
    double PSNR_channel[3], PSNR_overall;
    double SSIM_channel[3], SSIM_overall;
    double Color_MSE, Edge_preservation, Zipper_effect;
    std::string FileName;
};

ImageMetrics get_metrics(const Mat& raw_img, const Mat& new_img) {
    ImageMetrics result;

    // Crop images
    Mat cropped_raw = raw_img(Range(5, raw_img.rows - 5), Range(5, raw_img.cols - 5));
    Mat cropped_new = new_img(Range(5, new_img.rows - 5), Range(5, new_img.cols - 5));

    // Split images into channels
    vector<Mat> channels_raw, channels_new;
    split(cropped_raw, channels_raw);
    split(cropped_new, channels_new);

    // PSNR for each channel and overall
    for (int c = 0; c < 3; ++c) {
        result.PSNR_channel[c] = getPSNR(channels_raw[c], channels_new[c]);
    }
    result.PSNR_overall = getPSNR(cropped_raw, cropped_new);

    // SSIM for each channel and overall
    for (int c = 0; c < 3; ++c) {
        result.SSIM_channel[c] = getMSSIM(channels_raw[c], channels_new[c])[0];
    }
    result.SSIM_overall = getMSSIM(cropped_raw, cropped_new)[0];

    // Implement SSIM for overall image using getMSSIM directly (assuming data_range is 255)
    result.SSIM_overall = getMSSIM(cropped_raw, cropped_new)[0];

    // Color accuracy in LAB color space
    Mat lab_raw, lab_new;
    cvtColor(cropped_raw, lab_raw, COLOR_BGR2Lab);
    cvtColor(cropped_new, lab_new, COLOR_BGR2Lab);
    Mat diff = lab_raw - lab_new;
    result.Color_MSE = mean(diff.mul(diff))[0];

    // Edge preservation
    Mat gray_raw, gray_new, edges_raw, edges_new;
    cvtColor(cropped_raw, gray_raw, COLOR_BGR2GRAY);
    cvtColor(cropped_new, gray_new, COLOR_BGR2GRAY);
    Canny(gray_raw, edges_raw, 100, 200);
    Canny(gray_new, edges_new, 100, 200);
    result.Edge_preservation = countNonZero(edges_raw & edges_new) / (double)countNonZero(edges_raw | edges_new);

    // Zipper effect detection
    Mat laplacian;
    Laplacian(gray_new, laplacian, CV_64F);
    result.Zipper_effect = norm(laplacian, NORM_L2) / (laplacian.rows * laplacian.cols);

    return result;
}

void print_metrics(const vector<ImageMetrics>& metrics) {
    for (size_t i = 0; i < metrics.size(); ++i) {
        cout << "Image " << metrics[i].FileName << endl;
        cout << fixed << setprecision(2);
        cout << "PSNR Channels: " << metrics[i].PSNR_channel[0] << ", " << metrics[i].PSNR_channel[1] << ", " << metrics[i].PSNR_channel[2] << endl;
        cout << "PSNR Overall: " << metrics[i].PSNR_overall << endl;
        cout << "SSIM Channels: " << metrics[i].SSIM_channel[0] << ", " << metrics[i].SSIM_channel[1] << ", " << metrics[i].SSIM_channel[2] << endl;
        cout << "SSIM Overall: " << metrics[i].SSIM_overall << endl;
        cout << "Color MSE: " << metrics[i].Color_MSE << endl;
        cout << "Edge Preservation: " << metrics[i].Edge_preservation << endl;
        cout << "Zipper Effect: " << metrics[i].Zipper_effect << endl;
        cout << endl;
    }
}

void save_metrics_to_csv(const vector<ImageMetrics>& metrics, const string& filename) {
    ofstream csv_file(filename);
    if (!csv_file) {
        cerr << "Error opening CSV file" << endl;
        return;
    }

    csv_file << "Image_Index,PSNR_Channel_0,PSNR_Channel_1,PSNR_Channel_2,PSNR_Overall,SSIM_Channel_0,SSIM_Channel_1,SSIM_Channel_2,SSIM_Overall,Color_MSE,Edge_Preservation,Zipper_Effect\n";

    for (size_t i = 0; i < metrics.size(); ++i) {
        csv_file << metrics[i].FileName << ",";
        csv_file << fixed << setprecision(2);
        csv_file << metrics[i].PSNR_channel[0] << "," << metrics[i].PSNR_channel[1] << "," << metrics[i].PSNR_channel[2] << ",";
        csv_file << metrics[i].PSNR_overall << ",";
        csv_file << metrics[i].SSIM_channel[0] << "," << metrics[i].SSIM_channel[1] << "," << metrics[i].SSIM_channel[2] << ",";
        csv_file << metrics[i].SSIM_overall << ",";
        csv_file << metrics[i].Color_MSE << "," << metrics[i].Edge_preservation << "," << metrics[i].Zipper_effect << endl;
    }

    csv_file.close();
}


std::string get_filename(const std::string& full_path) {
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return full_path.substr(last_slash + 1);
    }
    else {
        return full_path;
    }
}

void save_image(const cv::Mat& image, const std::string& filename) {
    string folderPath = "Data/";
    string imagePath = folderPath + filename + ".png";
    string csvPath = folderPath + filename + ".csv";

    // Save image
    imwrite(imagePath, image);

    // Save data to CSV (assuming image is a Mat of CV_8UC3)
    /*ofstream csvFile(csvPath);
    if (csvFile.is_open()) {
        csvFile << format(image, cv::Formatter::FMT_CSV) << endl;
        csvFile.close();
    }
    else {
        cerr << "Error opening CSV file" << endl;
    }*/
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

    string input_folder = "Data/input/";
    string output_folder = "Data/";

    vector<string> image_files;
    vector<ImageMetrics> image_metrices;
    glob(input_folder + "*.*", image_files, false); // Adjust pattern as needed

    for (const string& filename : image_files) {
        string file_name = get_filename(filename);
        file_name = file_name.substr(0, file_name.size() - 4);

        Mat img = imread(filename, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Could not open or find the image: " << filename << endl;
            continue;
        }

        // Create resized images
        Mat img_half, img_quarter;
        resize(img, img_half, Size(img.cols / 2, img.rows / 2),0,0,INTER_CUBIC);
        resize(img, img_quarter, Size(img.cols / 4, img.rows / 4),0,0, INTER_CUBIC);

        // Process each image (original, half, quarter)
        for (auto& image : { img, img_half, img_quarter }) {
            string size_suffix = (image.cols == img.cols) ? "" : to_string(img.cols / image.cols);
            string output_prefix = output_folder + file_name + "_" + size_suffix;

            cvtColor(image, image, COLOR_BGR2RGB);

            Mat Bayer = make_bayer(image);
            // save_image(Bayer, output_prefix + "_Bayer.png");

            Mat Demosaiced = Demosaic_DLMMSE(Bayer);

            cvtColor(Demosaiced, Demosaiced, COLOR_RGB2BGR);

            string output_filename = output_prefix + "_DLMMSE.png";
            imwrite(output_filename, Demosaiced);

            output_filename = output_prefix + "_GT.png";
            cvtColor(image, image, COLOR_RGB2BGR);
            imwrite(output_filename, image);

            cout << "[Success] File Saved: " << output_filename << "\n";

            // Computing Image Metrics:
            ImageMetrics metrics = get_metrics(image, Demosaiced);
            metrics.FileName = output_filename;
            image_metrices.push_back(metrics);
        }
    }

    print_metrics(image_metrices);
    save_metrics_to_csv(image_metrices, "metrics.csv");

    return 0;
}
