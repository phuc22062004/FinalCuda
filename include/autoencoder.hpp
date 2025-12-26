#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "config.h"

struct Tensor3D {
    int C, H, W;
    std::vector<float> data;

    Tensor3D() : C(0), H(0), W(0) {}
    Tensor3D(int C, int H, int W) : C(C), H(H), W(W), data(C * H * W, 0.0f) {}
    
    inline float& at(int c, int h, int w) {
        return data[(c * H + h) * W + w];
    }

    inline const float& at(int c, int h, int w) const {
        return data[(c * H + h) * W + w];
    }
};

class AutoencoderCPU {
    public:
        AutoencoderCPU();
    
        Tensor3D forward(const Tensor3D& x);  
        Tensor3D decode(const Tensor3D& z);   
        
        float train_step(const Tensor3D& x, float learning_rate);
        float get_loss() const { return last_loss; }
        
        // Extract features from encoder bottleneck (128*8*8 = 8192 features)
        void extract_features(const float* input_chw, float* output_features);
        
        bool save_weights(const std::string& filepath) const;
        bool load_weights(const std::string& filepath);
    
    private:
        std::vector<float> conv1_w; 
        std::vector<float> conv1_b;
        std::vector<float> conv2_w;
        std::vector<float> conv2_b;
        
        std::vector<float> conv3_w;
        std::vector<float> conv3_b;
        std::vector<float> conv4_w;
        std::vector<float> conv4_b;
        std::vector<float> conv5_w;
        std::vector<float> conv5_b;
    
        Tensor3D conv2d(const Tensor3D& x, 
                        const std::vector<float>& w, 
                        const std::vector<float>& b,
                        int outC,
                        bool use_padding = true);
    
        Tensor3D relu(const Tensor3D& x);
        Tensor3D maxpool(const Tensor3D& x);
        Tensor3D upsample2x(const Tensor3D& x);
        
        Tensor3D relu_backward(const Tensor3D& grad_out, const Tensor3D& x);
        Tensor3D maxpool_backward(const Tensor3D& grad_out, const Tensor3D& x);
        Tensor3D upsample2x_backward(const Tensor3D& grad_out);
        Tensor3D conv2d_backward(const Tensor3D& grad_out,
                                 const Tensor3D& x,
                                 std::vector<float>& w_grad,
                                 std::vector<float>& b_grad,
                                 const std::vector<float>& w,
                                 int outC,
                                 bool use_padding);
        void update_weights(float learning_rate);
        
        std::vector<float> conv1_w_grad, conv1_b_grad;
        std::vector<float> conv2_w_grad, conv2_b_grad;
        std::vector<float> conv3_w_grad, conv3_b_grad;
        std::vector<float> conv4_w_grad, conv4_b_grad;
        std::vector<float> conv5_w_grad, conv5_b_grad;
        
        float last_loss = 0.0f;
        
        Tensor3D conv1_input, conv1_output, relu1_output;
        Tensor3D maxpool1_output;
        Tensor3D conv2_output, relu2_output;
        Tensor3D maxpool2_output;
        Tensor3D conv3_output, relu3_output;
        Tensor3D upsample1_output;
        Tensor3D conv4_output, relu4_output;
        Tensor3D upsample2_output;
        Tensor3D conv5_output;
    };
