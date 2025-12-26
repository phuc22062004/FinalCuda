#include "autoencoder.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

AutoencoderCPU::AutoencoderCPU() {

    conv1_w.resize(256 * 3 * 3 * 3);
    conv1_b.resize(256);
    
    conv2_w.resize(128 * 256 * 3 * 3);
    conv2_b.resize(128);
    
    conv3_w.resize(128 * 128 * 3 * 3);
    conv3_b.resize(128);
    
    conv4_w.resize(256 * 128 * 3 * 3);
    conv4_b.resize(256);
    
    conv5_w.resize(3 * 256 * 3 * 3);
    conv5_b.resize(3);
    
    conv1_w_grad.resize(256 * 3 * 3 * 3, 0.0f);
    conv1_b_grad.resize(256, 0.0f);
    conv2_w_grad.resize(128 * 256 * 3 * 3, 0.0f);
    conv2_b_grad.resize(128, 0.0f);
    conv3_w_grad.resize(128 * 128 * 3 * 3, 0.0f);
    conv3_b_grad.resize(128, 0.0f);
    conv4_w_grad.resize(256 * 128 * 3 * 3, 0.0f);
    conv4_b_grad.resize(256, 0.0f);
    conv5_w_grad.resize(3 * 256 * 3 * 3, 0.0f);
    conv5_b_grad.resize(3, 0.0f);
    
    for (auto& w : conv1_w) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : conv2_w) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : conv3_w) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : conv4_w) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : conv5_w) w = ((rand() % 100) / 500.0f - 0.1f);
    
    std::fill(conv1_b.begin(), conv1_b.end(), 0.0f);
    std::fill(conv2_b.begin(), conv2_b.end(), 0.0f);
    std::fill(conv3_b.begin(), conv3_b.end(), 0.0f);
    std::fill(conv4_b.begin(), conv4_b.end(), 0.0f);
    std::fill(conv5_b.begin(), conv5_b.end(), 0.0f);
}

Tensor3D AutoencoderCPU::conv2d(
    const Tensor3D& x,
    const std::vector<float>& w,
    const std::vector<float>& b,
    int outC,
    bool use_padding)
{
    int inC = x.C;
    int H = x.H, W = x.W;
    int k = 3;
    int pad = use_padding ? 1 : 0;
    
    Tensor3D out(outC, H + 2*pad - 2, W + 2*pad - 2);

    for (int oc = 0; oc < outC; oc++) {
        for (int ic = 0; ic < inC; ic++) {
            for (int i = 0; i < out.H; i++) {
                for (int j = 0; j < out.W; j++) {
                    float sum = 0.0f;

                    for (int ki = 0; ki < k; ki++) {
                        for (int kj = 0; kj < k; kj++) {
                            int x_h = i + ki - pad;
                            int x_w = j + kj - pad;
                            
                            float val = 0.0f;
                            if (x_h >= 0 && x_h < H && x_w >= 0 && x_w < W) {
                                val = x.at(ic, x_h, x_w);
                            }
                            
                            float weight = w[((oc * inC + ic) * k + ki) * k + kj];
                            sum += val * weight;
                        }
                    }

                    out.at(oc, i, j) += sum;
                }
            }
        }

        for (int i = 0; i < out.H; i++)
            for (int j = 0; j < out.W; j++)
                out.at(oc, i, j) += b[oc];
    }

    return out;
}

Tensor3D AutoencoderCPU::relu(const Tensor3D& x) {
    Tensor3D out = x;
    for (float& v : out.data) v = std::max(0.0f, v);
    return out;
}

Tensor3D AutoencoderCPU::maxpool(const Tensor3D& x) {
    int C = x.C, H = x.H, W = x.W;
    Tensor3D out(C, H / 2, W / 2);

    for (int c = 0; c < C; c++)
        for (int i = 0; i < out.H; i++)
            for (int j = 0; j < out.W; j++) {
                float m = x.at(c, 2 * i, 2 * j);
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                        m = std::max(m, x.at(c, 2 * i + a, 2 * j + b));
                out.at(c, i, j) = m;
            }

    return out;
}

Tensor3D AutoencoderCPU::upsample2x(const Tensor3D& x) {
    Tensor3D out(x.C, x.H * 2, x.W * 2);

    for (int c = 0; c < x.C; c++)
        for (int i = 0; i < x.H; i++)
            for (int j = 0; j < x.W; j++) {
                float v = x.at(c, i, j);
                out.at(c, 2 * i,     2 * j)     = v;
                out.at(c, 2 * i + 1, 2 * j)     = v;
                out.at(c, 2 * i,     2 * j + 1) = v;
                out.at(c, 2 * i + 1, 2 * j + 1) = v;
            }

    return out;
}

Tensor3D AutoencoderCPU::forward(const Tensor3D& x) {
    conv1_input = x;
    conv1_output = conv2d(x, conv1_w, conv1_b, 256, true);
    relu1_output = relu(conv1_output);
    
    maxpool1_output = maxpool(relu1_output);
    
    conv2_output = conv2d(maxpool1_output, conv2_w, conv2_b, 128, true);
    relu2_output = relu(conv2_output);
    
    maxpool2_output = maxpool(relu2_output);
    
    return maxpool2_output;
}

Tensor3D AutoencoderCPU::decode(const Tensor3D& z) {
    conv3_output = conv2d(z, conv3_w, conv3_b, 128, true);
    relu3_output = relu(conv3_output);
    
    upsample1_output = upsample2x(relu3_output);
    
    conv4_output = conv2d(upsample1_output, conv4_w, conv4_b, 256, true);
    relu4_output = relu(conv4_output);
    
    upsample2_output = upsample2x(relu4_output);
    
    conv5_output = conv2d(upsample2_output, conv5_w, conv5_b, 3, true);
    
    return conv5_output;
}

Tensor3D AutoencoderCPU::relu_backward(const Tensor3D& grad_out, const Tensor3D& x) {
    Tensor3D grad_in = grad_out;
    for (size_t i = 0; i < x.data.size(); i++) {
        if (x.data[i] <= 0.0f) {
            grad_in.data[i] = 0.0f;
        }
    }
    return grad_in;
}

Tensor3D AutoencoderCPU::maxpool_backward(const Tensor3D& grad_out, const Tensor3D& x) {
    Tensor3D grad_in(x.C, x.H, x.W);
    std::fill(grad_in.data.begin(), grad_in.data.end(), 0.0f);
    
    for (int c = 0; c < x.C; c++) {
        for (int i = 0; i < grad_out.H; i++) {
            for (int j = 0; j < grad_out.W; j++) {
                float max_val = x.at(c, 2 * i, 2 * j);
                int max_i = 2 * i, max_j = 2 * j;
                
                for (int a = 0; a < 2; a++) {
                    for (int b = 0; b < 2; b++) {
                        float val = x.at(c, 2 * i + a, 2 * j + b);
                        if (val > max_val) {
                            max_val = val;
                            max_i = 2 * i + a;
                            max_j = 2 * j + b;
                        }
                    }
                }
                
                grad_in.at(c, max_i, max_j) = grad_out.at(c, i, j);
            }
        }
    }
    
    return grad_in;
}

Tensor3D AutoencoderCPU::upsample2x_backward(const Tensor3D& grad_out) {
    Tensor3D grad_in(grad_out.C, grad_out.H / 2, grad_out.W / 2);
    
    for (int c = 0; c < grad_in.C; c++) {
        for (int i = 0; i < grad_in.H; i++) {
            for (int j = 0; j < grad_in.W; j++) {
                float sum = 0.0f;
                for (int a = 0; a < 2; a++) {
                    for (int b = 0; b < 2; b++) {
                        sum += grad_out.at(c, 2 * i + a, 2 * j + b);
                    }
                }
                grad_in.at(c, i, j) = sum;
            }
        }
    }
    
    return grad_in;
}

Tensor3D AutoencoderCPU::conv2d_backward(
    const Tensor3D& grad_out,
    const Tensor3D& x,
    std::vector<float>& w_grad,
    std::vector<float>& b_grad,
    const std::vector<float>& w,
    int outC,
    bool use_padding)
{
    int inC = x.C;
    int k = 3;
    int pad = use_padding ? 1 : 0;
    
    std::fill(w_grad.begin(), w_grad.end(), 0.0f);
    std::fill(b_grad.begin(), b_grad.end(), 0.0f);
    
    for (int oc = 0; oc < outC; oc++) {
        for (int ic = 0; ic < inC; ic++) {
            for (int ki = 0; ki < k; ki++) {
                for (int kj = 0; kj < k; kj++) {
                    float sum = 0.0f;
                    for (int i = 0; i < grad_out.H; i++) {
                        for (int j = 0; j < grad_out.W; j++) {
                            int x_h = i + ki - pad;
                            int x_w = j + kj - pad;
                            if (x_h >= 0 && x_h < x.H && x_w >= 0 && x_w < x.W) {
                                sum += grad_out.at(oc, i, j) * x.at(ic, x_h, x_w);
                            }
                        }
                    }
                    int idx = ((oc * inC + ic) * k + ki) * k + kj;
                    w_grad[idx] = sum;
                }
            }
        }
    }
    
    for (int oc = 0; oc < outC; oc++) {
        float sum = 0.0f;
        for (int i = 0; i < grad_out.H; i++) {
            for (int j = 0; j < grad_out.W; j++) {
                sum += grad_out.at(oc, i, j);
            }
        }
        b_grad[oc] = sum;
    }
    
    Tensor3D grad_in(inC, x.H, x.W);
    std::fill(grad_in.data.begin(), grad_in.data.end(), 0.0f);
    
    for (int ic = 0; ic < inC; ic++) {
        for (int i = 0; i < x.H; i++) {
            for (int j = 0; j < x.W; j++) {
                float sum = 0.0f;
                for (int oc = 0; oc < outC; oc++) {
                    for (int ki = 0; ki < k; ki++) {
                        for (int kj = 0; kj < k; kj++) {
                            int out_i = i - ki + pad;
                            int out_j = j - kj + pad;
                            if (out_i >= 0 && out_i < grad_out.H && out_j >= 0 && out_j < grad_out.W) {
                                int idx = ((oc * inC + ic) * k + ki) * k + kj;
                                sum += grad_out.at(oc, out_i, out_j) * w[idx];
                            }
                        }
                    }
                }
                grad_in.at(ic, i, j) = sum;
            }
        }
    }
    
    return grad_in;
}

void AutoencoderCPU::update_weights(float learning_rate) {
    for (size_t i = 0; i < conv1_w.size(); i++) {
        conv1_w[i] -= learning_rate * conv1_w_grad[i];
    }
    for (size_t i = 0; i < conv1_b.size(); i++) {
        conv1_b[i] -= learning_rate * conv1_b_grad[i];
    }
    
    for (size_t i = 0; i < conv2_w.size(); i++) {
        conv2_w[i] -= learning_rate * conv2_w_grad[i];
    }
    for (size_t i = 0; i < conv2_b.size(); i++) {
        conv2_b[i] -= learning_rate * conv2_b_grad[i];
    }
    
    for (size_t i = 0; i < conv3_w.size(); i++) {
        conv3_w[i] -= learning_rate * conv3_w_grad[i];
    }
    for (size_t i = 0; i < conv3_b.size(); i++) {
        conv3_b[i] -= learning_rate * conv3_b_grad[i];
    }
    
    for (size_t i = 0; i < conv4_w.size(); i++) {
        conv4_w[i] -= learning_rate * conv4_w_grad[i];
    }
    for (size_t i = 0; i < conv4_b.size(); i++) {
        conv4_b[i] -= learning_rate * conv4_b_grad[i];
    }
    
    for (size_t i = 0; i < conv5_w.size(); i++) {
        conv5_w[i] -= learning_rate * conv5_w_grad[i];
    }
    for (size_t i = 0; i < conv5_b.size(); i++) {
        conv5_b[i] -= learning_rate * conv5_b_grad[i];
    }
}

float AutoencoderCPU::train_step(const Tensor3D& x, float learning_rate) {
    Tensor3D latent = forward(x);
    Tensor3D reconstructed = decode(latent);
    
    Tensor3D grad_loss(conv5_output.C, conv5_output.H, conv5_output.W);
    float loss_sum = 0.0f;
    
    for (int c = 0; c < conv5_output.C; c++) {
        for (int h = 0; h < conv5_output.H; h++) {
            for (int w = 0; w < conv5_output.W; w++) {
                float pred = conv5_output.at(c, h, w);
                float tgt = x.at(c, h, w);
                float diff = pred - tgt;
                loss_sum += diff * diff;
                grad_loss.at(c, h, w) = 2.0f * diff / (conv5_output.C * conv5_output.H * conv5_output.W);
            }
        }
    }
    
    last_loss = loss_sum / (conv5_output.C * conv5_output.H * conv5_output.W);
    
    Tensor3D grad_conv5 = conv2d_backward(grad_loss, upsample2_output, conv5_w_grad, conv5_b_grad, conv5_w, 3, true);
    Tensor3D grad_upsample2 = upsample2x_backward(grad_conv5);
    Tensor3D grad_relu4 = relu_backward(grad_upsample2, conv4_output);
    Tensor3D grad_conv4 = conv2d_backward(grad_relu4, upsample1_output, conv4_w_grad, conv4_b_grad, conv4_w, 256, true);
    Tensor3D grad_upsample1 = upsample2x_backward(grad_conv4);
    Tensor3D grad_relu3 = relu_backward(grad_upsample1, conv3_output);
    Tensor3D grad_conv3 = conv2d_backward(grad_relu3, maxpool2_output, conv3_w_grad, conv3_b_grad, conv3_w, 128, true);
    
    Tensor3D grad_maxpool2 = maxpool_backward(grad_conv3, relu2_output);
    Tensor3D grad_relu2 = relu_backward(grad_maxpool2, conv2_output);
    Tensor3D grad_conv2 = conv2d_backward(grad_relu2, maxpool1_output, conv2_w_grad, conv2_b_grad, conv2_w, 128, true);
    Tensor3D grad_maxpool1 = maxpool_backward(grad_conv2, relu1_output);
    Tensor3D grad_relu1 = relu_backward(grad_maxpool1, conv1_output);
    Tensor3D grad_conv1 = conv2d_backward(grad_relu1, conv1_input, conv1_w_grad, conv1_b_grad, conv1_w, 256, true);
    
    update_weights(learning_rate);
    
    return last_loss;
}

bool AutoencoderCPU::save_weights(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.write((const char*)conv1_w.data(), conv1_w.size() * sizeof(float));
    file.write((const char*)conv1_b.data(), conv1_b.size() * sizeof(float));
    file.write((const char*)conv2_w.data(), conv2_w.size() * sizeof(float));
    file.write((const char*)conv2_b.data(), conv2_b.size() * sizeof(float));
    
    file.write((const char*)conv3_w.data(), conv3_w.size() * sizeof(float));
    file.write((const char*)conv3_b.data(), conv3_b.size() * sizeof(float));
    file.write((const char*)conv4_w.data(), conv4_w.size() * sizeof(float));
    file.write((const char*)conv4_b.data(), conv4_b.size() * sizeof(float));
    file.write((const char*)conv5_w.data(), conv5_w.size() * sizeof(float));
    file.write((const char*)conv5_b.data(), conv5_b.size() * sizeof(float));
    
    file.close();
    return true;
}

bool AutoencoderCPU::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.read((char*)conv1_w.data(), conv1_w.size() * sizeof(float));
    file.read((char*)conv1_b.data(), conv1_b.size() * sizeof(float));
    file.read((char*)conv2_w.data(), conv2_w.size() * sizeof(float));
    file.read((char*)conv2_b.data(), conv2_b.size() * sizeof(float));
    
    file.read((char*)conv3_w.data(), conv3_w.size() * sizeof(float));
    file.read((char*)conv3_b.data(), conv3_b.size() * sizeof(float));
    file.read((char*)conv4_w.data(), conv4_w.size() * sizeof(float));
    file.read((char*)conv4_b.data(), conv4_b.size() * sizeof(float));
    file.read((char*)conv5_w.data(), conv5_w.size() * sizeof(float));
    file.read((char*)conv5_b.data(), conv5_b.size() * sizeof(float));
    
    file.close();
    return true;
}

// Extract features from encoder bottleneck (128*8*8 = 8192 features)
void AutoencoderCPU::extract_features(const float* input_chw, float* output_features) {
    // Convert input to Tensor3D (3x32x32)
    Tensor3D x(3, 32, 32);
    for (int i = 0; i < 3072; i++) {
        x.data[i] = input_chw[i];
    }
    
    // Run encoder only (stop at bottleneck)
    Tensor3D conv1_out = conv2d(x, conv1_w, conv1_b, 256, true);
    Tensor3D relu1_out = relu(conv1_out);
    Tensor3D pool1_out = maxpool(relu1_out);
    
    Tensor3D conv2_out = conv2d(pool1_out, conv2_w, conv2_b, 128, true);
    Tensor3D relu2_out = relu(conv2_out);
    Tensor3D bottleneck = maxpool(relu2_out);  // 128x8x8 = 8192
    
    // Copy bottleneck features to output
    for (int i = 0; i < 8192; i++) {
        output_features[i] = bottleneck.data[i];
    }
}

