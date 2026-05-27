#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

static Logger gLogger;

struct BBox {
    float x, y, w, h;
};

// Load serialized TensorRT engine from file
nvinfer1::ICudaEngine* loadEngine(const std::string& enginePath, nvinfer1::IRuntime* runtime) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open engine file: " << enginePath << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
}

// Crop search/template region centered on target bbox, pad with black if out of bounds
cv::Mat sampleTarget(const cv::Mat& im, const BBox& targetBB, float searchAreaFactor,
                     int outputSz, float& resizeFactor) {
    float w = targetBB.w;
    float h = targetBB.h;
    float cropSz = std::ceil(std::sqrt(w * h) * searchAreaFactor);

    float cx = targetBB.x + 0.5f * w;
    float cy = targetBB.y + 0.5f * h;

    int x1 = static_cast<int>(std::round(cx - cropSz * 0.5f));
    int y1 = static_cast<int>(std::round(cy - cropSz * 0.5f));
    int x2 = x1 + static_cast<int>(cropSz);
    int y2 = y1 + static_cast<int>(cropSz);

    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(0, x2 - im.cols + 1);
    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(0, y2 - im.rows + 1);

    int crop_x1 = x1 + x1_pad;
    int crop_y1 = y1 + y1_pad;
    int crop_x2 = x2 - x2_pad;
    int crop_y2 = y2 - y2_pad;

    cv::Mat imCrop = im(cv::Rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1));

    cv::Mat imCropPadded;
    cv::copyMakeBorder(imCrop, imCropPadded, y1_pad, y2_pad, x1_pad, x2_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    resizeFactor = static_cast<float>(outputSz) / cropSz;

    cv::Mat resized;
    cv::resize(imCropPadded, resized, cv::Size(outputSz, outputSz));
    return resized;
}

// Preprocess: BGR->RGB, normalize, HWC->CHW, write to float buffer
void preprocess(const cv::Mat& img, float* buffer, int channels, int height, int width) {
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // OpenCV is BGR, model expects RGB: channel 0=R(2), 1=G(1), 2=B(0)
                int bgr_c = 2 - c;
                float pixel = static_cast<float>(img.at<cv::Vec3b>(h, w)[bgr_c]) / 255.0f;
                buffer[c * height * width + h * width + w] = (pixel - mean[c]) / stdv[c];
            }
        }
    }
}

BBox clipBox(const BBox& box, int H, int W, int margin = 10) {
    float x1 = box.x;
    float y1 = box.y;
    float x2 = x1 + box.w;
    float y2 = y1 + box.h;

    x1 = std::min(std::max(0.0f, x1), static_cast<float>(W - margin));
    x2 = std::min(std::max(static_cast<float>(margin), x2), static_cast<float>(W));
    y1 = std::min(std::max(0.0f, y1), static_cast<float>(H - margin));
    y2 = std::min(std::max(static_cast<float>(margin), y2), static_cast<float>(H));

    float nw = std::max(static_cast<float>(margin), x2 - x1);
    float nh = std::max(static_cast<float>(margin), y2 - y1);

    return {x1, y1, nw, nh};
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./sgla_tracker --engine <path> --video <path>" << std::endl;
        return -1;
    }

    std::string enginePath, videoPath;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--engine" && i + 1 < argc) enginePath = argv[++i];
        else if (arg == "--video" && i + 1 < argc) videoPath = argv[++i];
    }

    if (enginePath.empty() || videoPath.empty()) {
        std::cerr << "Both --engine and --video are required." << std::endl;
        return -1;
    }

    // TensorRT setup
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = loadEngine(enginePath, runtime);
    if (!engine) return -1;
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // Allocate GPU buffers
    const int templateC = 3, templateH = 128, templateW = 128;
    const int searchC = 3, searchH = 256, searchW = 256;
    const int outputSize = 4;

    size_t templateBytes = templateC * templateH * templateW * sizeof(float);
    size_t searchBytes = searchC * searchH * searchW * sizeof(float);
    size_t outputBytes = outputSize * sizeof(float);

    void* d_template;
    void* d_search;
    void* d_output;
    cudaMalloc(&d_template, templateBytes);
    cudaMalloc(&d_search, searchBytes);
    cudaMalloc(&d_output, outputBytes);

    int templateIdx = engine->getBindingIndex("template");
    int searchIdx = engine->getBindingIndex("search");
    int outputIdx = engine->getBindingIndex("pred_boxes");

    void* bindings[3];
    bindings[templateIdx] = d_template;
    bindings[searchIdx] = d_search;
    bindings[outputIdx] = d_output;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Host buffers
    std::vector<float> h_template(templateC * templateH * templateW);
    std::vector<float> h_search(searchC * searchH * searchW);
    std::vector<float> h_output(outputSize);

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << videoPath << std::endl;
        return -1;
    }

    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        std::cerr << "Cannot read first frame." << std::endl;
        return -1;
    }

    // Select initial ROI
    cv::Rect2d roi = cv::selectROI("Select Object", firstFrame, false);
    cv::destroyAllWindows();
    BBox state = {static_cast<float>(roi.x), static_cast<float>(roi.y),
                  static_cast<float>(roi.width), static_cast<float>(roi.height)};

    // Process template (once)
    const float templateFactor = 2.0f;
    const int templateSize = 128;
    const float searchFactor = 4.0f;
    const int searchSize = 256;

    float resizeFactorZ;
    cv::Mat firstFrameRgb;
    cv::cvtColor(firstFrame, firstFrameRgb, cv::COLOR_BGR2RGB);
    cv::Mat templatePatch = sampleTarget(firstFrameRgb, state, templateFactor, templateSize, resizeFactorZ);
    cv::cvtColor(templatePatch, templatePatch, cv::COLOR_RGB2BGR);
    preprocess(templatePatch, h_template.data(), templateC, templateH, templateW);

    // Upload template to GPU (only once)
    cudaMemcpyAsync(d_template, h_template.data(), templateBytes, cudaMemcpyHostToDevice, stream);

    // Tracking loop
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        int H = frame.rows;
        int W = frame.cols;

        // Crop search region
        float resizeFactor;
        cv::Mat frameRgb;
        cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);
        cv::Mat searchPatch = sampleTarget(frameRgb, state, searchFactor, searchSize, resizeFactor);
        cv::cvtColor(searchPatch, searchPatch, cv::COLOR_RGB2BGR);
        preprocess(searchPatch, h_search.data(), searchC, searchH, searchW);

        // Upload search to GPU
        cudaMemcpyAsync(d_search, h_search.data(), searchBytes, cudaMemcpyHostToDevice, stream);

        // Run inference
        context->enqueueV2(bindings, stream, nullptr);

        // Download output
        cudaMemcpyAsync(h_output.data(), d_output, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Postprocess: pred_boxes is (cx, cy, w, h) normalized
        float pred_cx = h_output[0] * searchSize / resizeFactor;
        float pred_cy = h_output[1] * searchSize / resizeFactor;
        float pred_w  = h_output[2] * searchSize / resizeFactor;
        float pred_h  = h_output[3] * searchSize / resizeFactor;

        // Map back to original image coordinates
        float cx_prev = state.x + 0.5f * state.w;
        float cy_prev = state.y + 0.5f * state.h;
        float halfSide = 0.5f * searchSize / resizeFactor;

        float cx_real = pred_cx + (cx_prev - halfSide);
        float cy_real = pred_cy + (cy_prev - halfSide);

        BBox newBox = {cx_real - 0.5f * pred_w, cy_real - 0.5f * pred_h, pred_w, pred_h};
        state = clipBox(newBox, H, W, 10);

        // Draw bbox
        int x1 = static_cast<int>(state.x);
        int y1 = static_cast<int>(state.y);
        int bw = static_cast<int>(state.w);
        int bh = static_cast<int>(state.h);
        cv::rectangle(frame, cv::Rect(x1, y1, bw, bh), cv::Scalar(0, 255, 0), 2);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_template);
    cudaFree(d_search);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
