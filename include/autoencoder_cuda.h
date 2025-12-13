#pragma once
#include <string>
#include <vector>

class AutoencoderCUDA {
public:
    AutoencoderCUDA();
    ~AutoencoderCUDA();

    // input: float[3072] theo layout CHW (C=3,H=32,W=32), đã normalize [0,1]
    float train_step(const float* input_chw, float learning_rate);

    bool save_weights(const std::string& filepath) const;
    bool load_weights(const std::string& filepath);
    float get_loss() const;

private:
    void forward();
    void backward();
    void update_weights(float learning_rate);

    // Host weight pointers (mutable for save_weights const method)
    mutable std::vector<float> h_conv1_w, h_conv1_b;
    mutable std::vector<float> h_conv2_w, h_conv2_b;
    mutable std::vector<float> h_conv3_w, h_conv3_b;
    mutable std::vector<float> h_conv4_w, h_conv4_b;
    mutable std::vector<float> h_conv5_w, h_conv5_b;
    
    // Device weight pointers
    float *d_conv1_w, *d_conv1_b;
    float *d_conv2_w, *d_conv2_b;
    float *d_conv3_w, *d_conv3_b;
    float *d_conv4_w, *d_conv4_b;
    float *d_conv5_w, *d_conv5_b;
    
    // Device gradient pointers
    float *d_conv1_w_grad, *d_conv1_b_grad;
    float *d_conv2_w_grad, *d_conv2_b_grad;
    float *d_conv3_w_grad, *d_conv3_b_grad;
    float *d_conv4_w_grad, *d_conv4_b_grad;
    float *d_conv5_w_grad, *d_conv5_b_grad;
    
    // Device activation buffers
    float *d_input;
    float *d_conv1_out, *d_relu1_out, *d_pool1_out;
    float *d_conv2_out, *d_relu2_out, *d_pool2_out;
    float *d_conv3_out, *d_relu3_out, *d_up1_out;
    float *d_conv4_out, *d_relu4_out, *d_up2_out;
    float *d_conv5_out;
    
    // Device gradient buffers
    float *d_grad_conv5, *d_grad_up2, *d_grad_relu4, *d_grad_conv4;
    float *d_grad_up1, *d_grad_relu3, *d_grad_conv3;
    float *d_grad_pool2, *d_grad_relu2, *d_grad_conv2;
    float *d_grad_pool1, *d_grad_relu1, *d_grad_conv1;
    
    float *d_loss;
    float last_loss;
};
