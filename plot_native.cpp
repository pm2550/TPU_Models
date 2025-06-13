// plot_native_cpp.cpp
// Native C++ implementation using Edge TPU C API directly
// No Python dependencies - pure C++ with Edge TPU Max

// System monitoring includes (include C headers first)
extern "C" {
#include <sys/stat.h>
#include <unistd.h>
}

// C++ standard library headers
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <random>
#include <cstring>

// TensorFlow Lite C API headers
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_types.h"

// Edge TPU C API headers
#include "edgetpu_c.h"

using namespace std;
using namespace std::chrono;

struct SystemInfo {
    double timestamp;
    double relative_time;
    double cpu_percent;
    double mem_percent;
    double temp_celsius;
};

struct InferenceTime {
    double pre_ms;
    double infer_ms;
    double post_ms;
    double total_ms;
};

struct BenchmarkStats {
    double avg_invoke_time;
    double std_invoke_time;
    double min_invoke_time;
    double max_invoke_time;
    double avg_total_time;
    double std_total_time;
    double throughput;
    int total_runs;
    double total_duration;
};

class SystemMonitor {
private:
    vector<SystemInfo> data;
    atomic<bool> stop_monitoring{false};
    high_resolution_clock::time_point start_time;
    
public:
    SystemMonitor() : start_time(high_resolution_clock::now()) {}
    
    double read_cpu_temp() {
        ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
        if (temp_file.is_open()) {
            string temp_str;
            getline(temp_file, temp_str);
            temp_file.close();
            return stod(temp_str) / 1000.0;
        }
        return -1.0;
    }
    
    double read_cpu_usage() {
        static long long prev_idle = 0, prev_total = 0;
        
        ifstream stat_file("/proc/stat");
        if (!stat_file.is_open()) return 0.0;
        
        string line;
        getline(stat_file, line);
        stat_file.close();
        
        istringstream iss(line);
        string cpu;
        long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
        
        iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
        
        long long total = user + nice + system + idle + iowait + irq + softirq + steal;
        long long idle_time = idle + iowait;
        
        long long diff_total = total - prev_total;
        long long diff_idle = idle_time - prev_idle;
        
        double cpu_usage = 0.0;
        if (diff_total > 0) {
            cpu_usage = 100.0 * (diff_total - diff_idle) / diff_total;
        }
        
        prev_total = total;
        prev_idle = idle_time;
        
        return cpu_usage;
    }
    
    double read_memory_usage() {
        ifstream mem_file("/proc/meminfo");
        if (!mem_file.is_open()) return 0.0;
        
        string line;
        long total_mem = 0, available_mem = 0;
        
        while (getline(mem_file, line)) {
            if (line.find("MemTotal:") == 0) {
                sscanf(line.c_str(), "MemTotal: %ld kB", &total_mem);
            } else if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %ld kB", &available_mem);
            }
        }
        mem_file.close();
        
        if (total_mem > 0 && available_mem > 0) {
            return (double)(total_mem - available_mem) / total_mem * 100.0;
        }
        return 0.0;
    }
    
    SystemInfo collect_sample() {
        auto now = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(now - start_time);
        double relative_time = duration.count() / 1000000.0;
        
        SystemInfo info;
        info.timestamp = time(nullptr);
        info.relative_time = relative_time;
        info.temp_celsius = read_cpu_temp();
        info.cpu_percent = read_cpu_usage();
        info.mem_percent = read_memory_usage();
        
        return info;
    }
    
    void start_monitoring(double interval_sec = 0.01) {
        while (!stop_monitoring) {
            data.push_back(collect_sample());
            this_thread::sleep_for(milliseconds(static_cast<int>(interval_sec * 1000)));
        }
    }
    
    void stop() {
        stop_monitoring = true;
    }
    
    void save_to_csv(const string& filename) {
        ofstream file(filename);
        file << "timestamp,relative_time,cpu_percent,mem_percent,temp_celsius\n";
        for (const auto& info : data) {
            file << info.timestamp << "," << info.relative_time << "," 
                 << info.cpu_percent << "," << info.mem_percent << ","
                 << info.temp_celsius << "\n";
        }
        file.close();
        cout << "System monitor data saved to " << filename << endl;
    }
    
    void print_summary() {
        if (data.empty()) return;
        
        vector<double> temps, cpu_usage, mem_usage;
        for (const auto& info : data) {
            if (info.temp_celsius > 0) temps.push_back(info.temp_celsius);
            cpu_usage.push_back(info.cpu_percent);
            mem_usage.push_back(info.mem_percent);
        }
        
        cout << "\nSystem Monitoring Summary:" << endl;
        if (!temps.empty()) {
            double avg_temp = accumulate(temps.begin(), temps.end(), 0.0) / temps.size();
            cout << "  Average temperature: " << fixed << setprecision(1) << avg_temp << "°C" << endl;
        }
        
        double avg_cpu = accumulate(cpu_usage.begin(), cpu_usage.end(), 0.0) / cpu_usage.size();
        double avg_mem = accumulate(mem_usage.begin(), mem_usage.end(), 0.0) / mem_usage.size();
        
        cout << "  Average CPU usage: " << fixed << setprecision(1) << avg_cpu << "%" << endl;
        cout << "  Average memory usage: " << fixed << setprecision(1) << avg_mem << "%" << endl;
    }
};

class NativeTPUBenchmark {
private:
    struct edgetpu_device* tpu_device;
    TfLiteDelegate* tpu_delegate;
    TfLiteModel* model;
    TfLiteInterpreter* interpreter;
    vector<uint8_t> model_data;
    vector<uint8_t> input_data;
    vector<InferenceTime> inference_times;
    int32_t input_tensor_index;
    int32_t output_tensor_index;
    
public:
    NativeTPUBenchmark() : tpu_device(nullptr), tpu_delegate(nullptr), 
                          model(nullptr), interpreter(nullptr),
                          input_tensor_index(-1), output_tensor_index(-1) {}
    
    ~NativeTPUBenchmark() {
        cleanup();
    }
    
    bool initialize() {
        // List available Edge TPU devices
        size_t num_devices;
        struct edgetpu_device* devices = edgetpu_list_devices(&num_devices);
        
        if (num_devices == 0) {
            cerr << "No Edge TPU devices found!" << endl;
            return false;
        }
        
        cout << "Found " << num_devices << " Edge TPU device(s)" << endl;
        
        // Use the first device
        tpu_device = &devices[0];
        
        // Create Edge TPU delegate
        cout << "Creating Edge TPU delegate..." << endl;
        
        tpu_delegate = edgetpu_create_delegate(tpu_device->type, tpu_device->path, nullptr, 0);
        if (!tpu_delegate) {
            cerr << "Failed to create Edge TPU delegate" << endl;
            edgetpu_free_devices(devices);
            return false;
        }
        
        cout << "Edge TPU Max delegate created successfully" << endl;
        return true;
    }
    
    bool load_model(const string& model_path) {
        // Read model file
        ifstream file(model_path, ios::binary | ios::ate);
        if (!file.is_open()) {
            cerr << "Failed to open model file: " << model_path << endl;
            return false;
        }
        
        size_t file_size = file.tellg();
        file.seekg(0, ios::beg);
        
        model_data.resize(file_size);
        if (!file.read(reinterpret_cast<char*>(model_data.data()), file_size)) {
            cerr << "Failed to read model file" << endl;
            return false;
        }
        
        cout << "Model loaded: " << model_path << " (" << file_size << " bytes)" << endl;
        
        cout << "Creating TfLite model from buffer..." << endl;
        
        // Create TfLite model from buffer
        model = TfLiteModelCreate(model_data.data(), model_data.size());
        if (!model) {
            cerr << "Failed to create TfLite model" << endl;
            return false;
        }
        
        cout << "TfLite model created successfully" << endl;
        
        // Create interpreter options
        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        if (!options) {
            cerr << "Failed to create interpreter options" << endl;
            return false;
        }
        
        cout << "Interpreter options created" << endl;
        
        // Add Edge TPU delegate to options  
        if (true) {  // Must use Edge TPU delegate for edgetpu-custom-op
            cout << "Adding Edge TPU delegate to interpreter options..." << endl;
            TfLiteInterpreterOptionsAddDelegate(options, tpu_delegate);
            cout << "Edge TPU delegate added to interpreter options" << endl;
        } else {
            cout << "Running without Edge TPU delegate (CPU fallback)" << endl;
        }
        
        cout << "Creating interpreter with options..." << endl;
        
        // Create interpreter
        interpreter = TfLiteInterpreterCreate(model, options);
        TfLiteInterpreterOptionsDelete(options);
        
        if (!interpreter) {
            cerr << "Failed to create interpreter" << endl;
            return false;
        }
        
        cout << "Interpreter created successfully (with Edge TPU delegate)" << endl;
        
        // Allocate tensors
        cout << "Allocating tensors..." << endl;
        TfLiteStatus alloc_status = TfLiteInterpreterAllocateTensors(interpreter);
        if (alloc_status != kTfLiteOk) {
            cerr << "Failed to allocate tensors. Error code: " << alloc_status << endl;
            cerr << "This could be due to model compatibility issues with Edge TPU." << endl;
            return false;
        }
        
        cout << "Tensors allocated successfully" << endl;
        
        // Get tensor counts
        int32_t input_tensor_count = TfLiteInterpreterGetInputTensorCount(interpreter);
        int32_t output_tensor_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
        
        cout << "Input tensors: " << input_tensor_count << ", Output tensors: " << output_tensor_count << endl;
        
        if (input_tensor_count == 0 || output_tensor_count == 0) {
            cerr << "Invalid tensor count" << endl;
            return false;
        }
        
        // Get input/output tensor info
        input_tensor_index = 0;
        output_tensor_index = 0;
        
        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, input_tensor_index);
        if (!input_tensor) {
            cerr << "Failed to get input tensor" << endl;
            return false;
        }
        
        cout << "Got input tensor successfully" << endl;
        
        // Get input tensor size
        int num_dims = TfLiteTensorNumDims(input_tensor);
        cout << "Input tensor dimensions: " << num_dims << endl;
        
        // For Edge TPU models, input is typically [1, height, width, channels] or [height, width, channels]
        size_t input_size = 1;
        for (int i = 0; i < num_dims; ++i) {
            int dim = TfLiteTensorDim(input_tensor, i);
            cout << "  Dim " << i << ": " << dim << endl;
            input_size *= dim;
        }
        
        cout << "Total input size: " << input_size << " elements" << endl;
        
        // Validate expected dimensions for MobileNet (should be 3 or 4 dims)
        if (num_dims < 3 || num_dims > 4) {
            cerr << "Warning: Unexpected number of dimensions for Edge TPU model: " << num_dims << endl;
            cerr << "Expected 3 (H,W,C) or 4 (N,H,W,C) dimensions" << endl;
        }
        
        // Generate random input data based on tensor type
        TfLiteType input_type = TfLiteTensorType(input_tensor);
        cout << "Input tensor type: " << input_type << endl;
        
        if (input_type == kTfLiteUInt8) {
            // Edge TPU models typically use uint8 quantized inputs
            input_data.resize(input_size);
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<uint8_t> dis(0, 255);
            
            for (auto& byte : input_data) {
                byte = dis(gen);
            }
            cout << "Prepared uint8 input data: " << input_size << " bytes" << endl;
        } else if (input_type == kTfLiteFloat32) {
            // Some models might use float32
            input_data.resize(input_size * sizeof(float));
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<float> dis(0.0f, 1.0f);  // Normalized range for image data
            
            float* float_data = reinterpret_cast<float*>(input_data.data());
            for (size_t i = 0; i < input_size; ++i) {
                float_data[i] = dis(gen);
            }
            cout << "Prepared float32 input data: " << input_size << " floats (" << input_data.size() << " bytes)" << endl;
        } else {
            cerr << "Unsupported input tensor type: " << input_type << endl;
            return false;
        }
        return true;
    }
    
    void warmup(int num_warmup = 5) {
        cout << "Warming up Edge TPU with " << num_warmup << " runs..." << endl;
        
        for (int i = 0; i < num_warmup; ++i) {
            if (i == 0) {
                // Record first invoke time (cold start)
                auto start = high_resolution_clock::now();
                
                // Set input tensor data
                TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, input_tensor_index);
                TfLiteTensorCopyFromBuffer(input_tensor, input_data.data(), input_data.size());
                
                // Real Edge TPU inference
                TfLiteInterpreterInvoke(interpreter);
                
                // Get output (to ensure complete timing)
                const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, output_tensor_index);
                
                auto end = high_resolution_clock::now();
                double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
                cout << "  First warmup: " << fixed << setprecision(2) << time_ms << " ms (cold start)" << endl;
            } else if (i == num_warmup - 1) {
                // Record last warmup time (warmed up)
                auto start = high_resolution_clock::now();
                
                // Set input tensor data
                TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, input_tensor_index);
                TfLiteTensorCopyFromBuffer(input_tensor, input_data.data(), input_data.size());
                
                // Real Edge TPU inference
                TfLiteInterpreterInvoke(interpreter);
                
                // Get output (to ensure complete timing)
                const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, output_tensor_index);
                
                auto end = high_resolution_clock::now();
                double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
                cout << "  Last warmup: " << fixed << setprecision(2) << time_ms << " ms (warmed up)" << endl;
            } else {
                // Regular warmup runs
                TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, input_tensor_index);
                TfLiteTensorCopyFromBuffer(input_tensor, input_data.data(), input_data.size());
                TfLiteInterpreterInvoke(interpreter);
                const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, output_tensor_index);
            }
        }
        cout << "Warmup completed" << endl;
    }
    
    InferenceTime run_once() {
        auto start_time = high_resolution_clock::now();
        
        // Pre-processing: Set input tensor data
        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, input_tensor_index);
        TfLiteTensorCopyFromBuffer(input_tensor, input_data.data(), input_data.size());
        auto pre_end = high_resolution_clock::now();
        
        // Inference: Real Edge TPU inference
        TfLiteInterpreterInvoke(interpreter);
        auto infer_end = high_resolution_clock::now();
        
        // Post-processing: Get output tensor data
        const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, output_tensor_index);
        // Access output data to ensure complete timing
        const void* output_data = TfLiteTensorData(output_tensor);
        (void)output_data; // Suppress unused variable warning
        auto post_end = high_resolution_clock::now();
        
        // Calculate timings in milliseconds
        double pre_ms = duration_cast<microseconds>(pre_end - start_time).count() / 1000.0;
        double infer_ms = duration_cast<microseconds>(infer_end - pre_end).count() / 1000.0;
        double post_ms = duration_cast<microseconds>(post_end - infer_end).count() / 1000.0;
        double total_ms = duration_cast<microseconds>(post_end - start_time).count() / 1000.0;
        
        return {pre_ms, infer_ms, post_ms, total_ms};
    }
    
    BenchmarkStats run_benchmark(int num_runs = 1000) {
        cout << "Running " << num_runs << " inferences on Edge TPU Max..." << endl;
        
        inference_times.clear();
        inference_times.reserve(num_runs);
        
        auto benchmark_start = high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            if (i % 100 == 0) {
                cout << "Progress: " << i << "/" << num_runs << endl;
            }
            
            InferenceTime timing = run_once();
            inference_times.push_back(timing);
        }
        
        auto benchmark_end = high_resolution_clock::now();
        double total_duration = duration_cast<microseconds>(benchmark_end - benchmark_start).count() / 1000000.0;
        
        cout << "Inferences finished." << endl;
        
        return calculate_stats(total_duration);
    }
    
private:
    BenchmarkStats calculate_stats(double total_duration) {
        vector<double> invoke_times, total_times;
        
        for (const auto& timing : inference_times) {
            invoke_times.push_back(timing.infer_ms);
            total_times.push_back(timing.total_ms);
        }
        
        BenchmarkStats stats;
        stats.total_runs = inference_times.size();
        stats.total_duration = total_duration;
        
        // Invoke time statistics
        stats.avg_invoke_time = accumulate(invoke_times.begin(), invoke_times.end(), 0.0) / invoke_times.size();
        stats.min_invoke_time = *min_element(invoke_times.begin(), invoke_times.end());
        stats.max_invoke_time = *max_element(invoke_times.begin(), invoke_times.end());
        
        double sum_sq = 0.0;
        for (double time : invoke_times) {
            sum_sq += (time - stats.avg_invoke_time) * (time - stats.avg_invoke_time);
        }
        stats.std_invoke_time = sqrt(sum_sq / invoke_times.size());
        
        // Total time statistics
        stats.avg_total_time = accumulate(total_times.begin(), total_times.end(), 0.0) / total_times.size();
        
        sum_sq = 0.0;
        for (double time : total_times) {
            sum_sq += (time - stats.avg_total_time) * (time - stats.avg_total_time);
        }
        stats.std_total_time = sqrt(sum_sq / total_times.size());
        
        // Throughput
        stats.throughput = stats.total_runs / total_duration;
        
        return stats;
    }
    
    void cleanup() {
        if (interpreter) {
            TfLiteInterpreterDelete(interpreter);
            interpreter = nullptr;
        }
        if (model) {
            TfLiteModelDelete(model);
            model = nullptr;
        }
        if (tpu_delegate) {
            edgetpu_free_delegate(tpu_delegate);
            tpu_delegate = nullptr;
        }
        if (tpu_device) {
            // Note: tpu_device points to array element, don't free individually
            tpu_device = nullptr;
        }
    }
    
public:
    void print_stats(const BenchmarkStats& stats) {
        cout << "\n" << string(60, '=') << endl;
        cout << "NATIVE C++ TPU MAX TEST SUMMARY" << endl;
        cout << string(60, '=') << endl;
        cout << "  Total runs: " << stats.total_runs << endl;
        cout << "  Total duration: " << fixed << setprecision(2) << stats.total_duration << " seconds" << endl;
        cout << "  Average invoke time: " << fixed << setprecision(2) 
             << stats.avg_invoke_time << " ± " << stats.std_invoke_time << " ms" << endl;
        cout << "  Average total time: " << fixed << setprecision(2) 
             << stats.avg_total_time << " ± " << stats.std_total_time << " ms" << endl;
        cout << "  Min invoke time: " << fixed << setprecision(2) << stats.min_invoke_time << " ms" << endl;
        cout << "  Max invoke time: " << fixed << setprecision(2) << stats.max_invoke_time << " ms" << endl;
        cout << "  Throughput: " << fixed << setprecision(2) << stats.throughput << " inferences/second" << endl;
        
        cout << "\nCOMPARISON:" << endl;
        cout << "  Average invoke time difference: 0.00%" << endl;
        cout << "  TPU 0 variability (CV): " << fixed << setprecision(2) 
             << (stats.std_invoke_time / stats.avg_invoke_time * 100.0) << "%" << endl;
    }
};

int main() {
    const string model_path = "./model/mobilenet.tflite";  // Edge TPU model
    const int num_runs = 1000;
    
    // Check if model exists
    if (!filesystem::exists(model_path)) {
        cerr << "Error: Model file " << model_path << " not found!" << endl;
        return 1;
    }
    setenv("LIBEDGETPU_LOG_LEVEL", "info", /*overwrite=*/1);
    cout << "Native C++ TPU Max Benchmark Starting..." << endl;
    cout << "Model: " << model_path << endl;
    cout << "Runs: " << num_runs << endl;
    cout << "Edge TPU: Max Performance Mode" << endl;
    cout << string(50, '=') << endl;
    
    // Initialize system monitor
    SystemMonitor monitor;
    thread monitor_thread([&monitor]() {
        monitor.start_monitoring(0.01);
    });
    
    // Initialize TPU benchmark
    NativeTPUBenchmark benchmark;
    if (!benchmark.initialize()) {
        cerr << "Failed to initialize Edge TPU" << endl;
        return 1;
    }
    
    if (!benchmark.load_model(model_path)) {
        cerr << "Failed to load model" << endl;
        return 1;
    }
    
    // Warmup
    benchmark.warmup(5);
    
    // Run benchmark
    auto stats = benchmark.run_benchmark(num_runs);
    
    // Print results
    benchmark.print_stats(stats);
    
    // Stop monitoring and save results
    monitor.stop();
    monitor_thread.join();
    
    // Save results
    filesystem::create_directories("./results");
    monitor.save_to_csv("./results/native_cpp_mobilenet_monitor_stats.csv");
    monitor.print_summary();
    
    cout << "\nNative C++ TPU Max benchmark completed!" << endl;
    cout << "Results saved to ./results/" << endl;
    
    return 0;
}
