// =============================================================================
// profile_spmv_gpu.cpp
// =============================================================================
// GPU SpMV profiling executable for occupancy and memory bandwidth measurement.
//
// What this profiles
// ------------------
// 1. Achieved occupancy % per kernel launch (via nvprof metrics)
// 2. Effective memory bandwidth (GB/s) for SpMV operation
// 3. L1/L2 cache hit rates (where available)
//
// Profiling Approach
// ------------------
// - Uses NVTX markers for visual profiling in nvprof/nsys
// - Measures kernel execution time with CUDA events
// - Calculates memory bandwidth from: (bytes_read + bytes_written) / time
// - CSV output for post-processing and reporting
//
// Usage
// -----
//   nvprof --metrics achieved_occupancy,global_load_throughput,\
//           l1_cache_global_hit_rate ./profile_spmv_gpu --matrix path/to/matrix.mtx
//
//   Or with NVTX markers:
//   ./profile_spmv_gpu --matrix path/to/matrix.mtx
//
//   Output: results/profiling_report.csv
//
// Exit codes: 0 = success, 1 = error
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include "spmv_cpu.h"
#include "timer.h"
#include "matrix_market.h"
#include "gpu_utils.h"

namespace {

// =============================================================================
// ProfilingMetrics — collected per kernel launch
// =============================================================================
struct ProfilingMetrics {
    std::string kernel_name;
    std::string matrix_name;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    // Timing
    double time_ms = 0.0;

    // Memory bandwidth (GB/s)
    double bandwidth_gbs = 0.0;

    // Theoretical peak bandwidth for Ampere A100: ~2TB/s HBM
    // For RTX 3080: ~760 GB/s
    static constexpr double RTX_3080_PEAK_BW = 760.0;  // GB/s

    // Compute memory traffic for CSR SpMV:
    //   - Read: nnz values (8 bytes) + nnz col_index (8 bytes) + (rows+1) row_ptr (8 bytes)
    //   - Read: cols elements of x vector (8 bytes each)
    //   - Write: rows elements of y vector (8 bytes each)
    // Total bytes = 8*(2*nnz + rows + 1 + cols + rows) = 8*(2*nnz + cols + 2*rows + 1)
    void compute_bandwidth() {
        const double bytes_read = 8.0 * (nnz + nnz + (rows + 1) + cols);  // values, col_index, row_ptr, x
        const double bytes_written = 8.0 * rows;  // y vector
        const double total_bytes = bytes_read + bytes_written;
        bandwidth_gbs = (total_bytes / 1e9) / (time_ms / 1e3);
    }
};

// =============================================================================
// NVTX Range helpers
// =============================================================================
class NVTXRange {
    const char* name_;
public:
    explicit NVTXRange(const char* name) : name_(name) {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = 0xFF00FF00;  // Green
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name_;
        nvtxMarkEx(&eventAttrib);
    }
};

// =============================================================================
// print_metrics — print metrics to stdout
// =============================================================================
void print_metrics(const ProfilingMetrics& m) {
    std::cout << "  Kernel:        " << m.kernel_name << "\n";
    std::cout << "  Matrix:        " << m.matrix_name << " (" << m.rows << "x" << m.cols << ", nnz=" << m.nnz << ")\n";
    std::cout << "  Time:          " << m.time_ms << " ms\n";
    std::cout << "  Bandwidth:     " << m.bandwidth_gbs << " GB/s\n";
    std::cout << "  Theoretical %: " << (m.bandwidth_gbs / ProfilingMetrics::RTX_3080_PEAK_BW * 100) << "%\n";
}

// =============================================================================
// write_csv_header — write CSV header to output file
// =============================================================================
void write_csv_header(std::ofstream& out) {
    out << "matrix_name,kernel_name,rows,cols,nnz,time_ms,bandwidth_gbs,theoretical_pct\n";
}

// =============================================================================
// write_csv_row — append metrics as CSV row
// =============================================================================
void write_csv_row(std::ofstream& out, const ProfilingMetrics& m) {
    out << std::quoted(m.matrix_name) << ","
        << std::quoted(m.kernel_name) << ","
        << m.rows << ","
        << m.cols << ","
        << m.nnz << ","
        << std::fixed << std::setprecision(6) << m.time_ms << ","
        << std::fixed << std::setprecision(4) << m.bandwidth_gbs << ","
        << std::fixed << std::setprecision(2) << (m.bandwidth_gbs / ProfilingMetrics::RTX_3080_PEAK_BW * 100) << "\n";
}

// =============================================================================
// get_matrix_name — extract filename without path
// =============================================================================
std::string get_matrix_name(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

// =============================================================================
// run_v1_kernel — profile GPU SpMV v1 kernel
// =============================================================================
ProfilingMetrics run_v1_kernel(const spmv::SparseMatrix& A,
                               const spmv::DenseVector& x,
                               const std::string& matrix_name) {
    ProfilingMetrics m;
    m.kernel_name = "spmv_gpu_v1";
    m.matrix_name = matrix_name;
    m.rows = A.rows;
    m.cols = A.cols;
    m.nnz = A.nnz;

    // Allocate device memory
    spmv::DeviceMatrix d_A = spmv::allocate_device_matrix(A);
    spmv::DeviceVector d_x = spmv::copy_vector_to_device(x);
    spmv::DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    // Kernel configuration: one thread per row
    constexpr int block_dim = 256;
    const int grid_dim = static_cast<int>((A.rows + block_dim - 1) / block_dim);

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // Warm-up run (not timed)
    {
        NVTXRange range("spmv_gpu_v1_warmup");
        spmv::spmv_gpu_v1_kernel<<<grid_dim, block_dim>>>(
            d_A.d_values, d_A.d_col_index, d_A.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    // Timed run with NVTX marker
    {
        NVTXRange range("spmv_gpu_v1");
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        spmv::spmv_gpu_v1_kernel<<<grid_dim, block_dim>>>(
            d_A.d_values, d_A.d_col_index, d_A.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    // Get elapsed time
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    m.time_ms = static_cast<double>(elapsed_ms);
    m.compute_bandwidth();

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    spmv::free_device_matrix(d_A);
    spmv::free_device_vector(d_x);
    spmv::free_device_vector(d_y);

    return m;
}

// =============================================================================
// run_v2_kernel — profile GPU SpMV v2 kernel (shared memory tiled)
// =============================================================================
ProfilingMetrics run_v2_kernel(const spmv::SparseMatrix& A,
                               const spmv::DenseVector& x,
                               const std::string& matrix_name) {
    ProfilingMetrics m;
    m.kernel_name = "spmv_gpu_v2";
    m.matrix_name = matrix_name;
    m.rows = A.rows;
    m.cols = A.cols;
    m.nnz = A.nnz;

    // Allocate device memory
    spmv::DeviceMatrix d_A = spmv::allocate_device_matrix(A);
    spmv::DeviceVector d_x = spmv::copy_vector_to_device(x);
    spmv::DeviceVector d_y;
    d_y.size = A.rows;
    CUDA_CHECK(cudaMalloc(&d_y.d_data, A.rows * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y.d_data, 0, A.rows * sizeof(double)));

    // Kernel configuration
    constexpr int block_dim = 256;
    constexpr int shared_mem_bytes = 32 * 1024;  // 32 KB
    const int grid_dim = static_cast<int>((A.rows + block_dim - 1) / block_dim);

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // Warm-up run (not timed)
    {
        NVTXRange range("spmv_gpu_v2_warmup");
        spmv::spmv_gpu_v2_kernel<4096><<<grid_dim, block_dim, shared_mem_bytes>>>(
            d_A.d_values, d_A.d_col_index, d_A.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    // Timed run with NVTX marker
    {
        NVTXRange range("spmv_gpu_v2");
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        spmv::spmv_gpu_v2_kernel<4096><<<grid_dim, block_dim, shared_mem_bytes>>>(
            d_A.d_values, d_A.d_col_index, d_A.d_row_ptr,
            d_x.d_data, d_y.d_data, A.rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    // Get elapsed time
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    m.time_ms = static_cast<double>(elapsed_ms);
    m.compute_bandwidth();

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    spmv::free_device_matrix(d_A);
    spmv::free_device_vector(d_x);
    spmv::free_device_vector(d_y);

    return m;
}

}  // anonymous namespace

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cout << std::scientific << std::setprecision(6);

    // -------------------------------------------------------------------------
    // Parse command line arguments
    // -------------------------------------------------------------------------
    std::string matrix_path;
    std::string output_csv = "results/profiling_report.csv";
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            matrix_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --matrix <path> [options]\n";
            std::cout << "  --matrix <path>   Path to Matrix Market file (.mtx) [required]\n";
            std::cout << "  --output <path>   CSV output file [default: results/profiling_report.csv]\n";
            std::cout << "  --verbose, -v     Verbose output\n";
            std::cout << "\nProfiling metrics collected:\n";
            std::cout << "  - Achieved occupancy % (via nvprof)\n";
            std::cout << "  - Memory bandwidth (GB/s)\n";
            std::cout << "  - L1/L2 cache hit rates\n";
            std::cout << "\nExample usage with nvprof:\n";
            std::cout << "  nvprof --metrics achieved_occupancy,global_load_throughput,\\\n";
            std::cout << "          l1_cache_global_hit_rate ./" << argv[0] << " \\\n";
            std::cout << "          --matrix ../data/bcspwr01.mtx\n";
            return 0;
        }
    }

    if (matrix_path.empty()) {
        std::cerr << "Error: --matrix <path> is required\n";
        std::cerr << "Usage: " << argv[0] << " --matrix <path> [--output results/profiling_report.csv]\n";
        return 1;
    }

    // -------------------------------------------------------------------------
    // Print header
    // -------------------------------------------------------------------------
    std::cout << "=======================================================\n";
    std::cout << "       GPU SpMV Profiling (Occupancy & Bandwidth)\n";
    std::cout << "=======================================================\n\n";

    // -------------------------------------------------------------------------
    // Load matrix
    // -------------------------------------------------------------------------
    std::string matrix_name = get_matrix_name(matrix_path);
    std::cout << "Loading matrix: " << matrix_name << "\n";
    std::cout << "Path: " << matrix_path << "\n\n";

    spmv::SparseMatrix A;
    try {
        A = spmv::parse_matrix_market(matrix_path);
    } catch (const std::exception& e) {
        std::cerr << "ERROR parsing matrix: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Matrix: " << A.rows << " x " << A.cols
              << ", nnz = " << A.nnz << "\n";
    std::cout << "Memory: " << A.memory_bytes() << " bytes\n\n";

    // -------------------------------------------------------------------------
    // Build input vector (all ones)
    // -------------------------------------------------------------------------
    spmv::DenseVector x(A.cols);
    spmv::fill_constant(x, 1.0);

    // -------------------------------------------------------------------------
    // Run v1 profiling
    // -------------------------------------------------------------------------
    std::cout << "Profiling spmv_gpu_v1...\n";
    ProfilingMetrics m_v1 = run_v1_kernel(A, x, matrix_name);
    if (verbose) {
        print_metrics(m_v1);
    } else {
        std::cout << "  Time: " << m_v1.time_ms << " ms, Bandwidth: " << m_v1.bandwidth_gbs << " GB/s\n";
    }

    // -------------------------------------------------------------------------
    // Run v2 profiling
    // -------------------------------------------------------------------------
    std::cout << "\nProfiling spmv_gpu_v2...\n";
    ProfilingMetrics m_v2 = run_v2_kernel(A, x, matrix_name);
    if (verbose) {
        print_metrics(m_v2);
    } else {
        std::cout << "  Time: " << m_v2.time_ms << " ms, Bandwidth: " << m_v2.bandwidth_gbs << " GB/s\n";
    }

    // -------------------------------------------------------------------------
    // Write CSV report
    // -------------------------------------------------------------------------
    // Create results directory if needed
    std::string output_dir = output_csv.substr(0, output_csv.find_last_of("/\\"));
    if (!output_dir.empty()) {
        std::string cmd = "mkdir -p " + output_dir;
        std::system(cmd.c_str());
    }

    std::ofstream csv_out;
    csv_out.open(output_csv, std::ios::app);  // Append mode
    if (!csv_out.is_open()) {
        std::cerr << "Error: Could not open CSV output: " << output_csv << "\n";
        return 1;
    }

    // Check if file is empty (need header)
    std::ifstream check_file(output_csv);
    bool need_header = check_file.peek() == std::ifstream::traits_type::eof();
    check_file.close();

    if (need_header) {
        write_csv_header(csv_out);
    }

    write_csv_row(csv_out, m_v1);
    write_csv_row(csv_out, m_v2);
    csv_out.close();

    std::cout << "\n=======================================================\n";
    std::cout << "            Profiling Results Summary\n";
    std::cout << "=======================================================\n\n";

    std::cout << std::left << std::setw(16) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(16) << "Bandwidth (GB/s)"
              << std::setw(14) << "Theoretical %\n";
    std::cout << std::string(58, '-') << "\n";
    std::cout << std::left << std::setw(16) << m_v1.kernel_name
              << std::right << std::setw(12) << std::fixed << std::setprecision(4) << m_v1.time_ms
              << std::setw(16) << std::fixed << std::setprecision(4) << m_v1.bandwidth_gbs
              << std::setw(14) << std::fixed << std::setprecision(2) << (m_v1.bandwidth_gbs / ProfilingMetrics::RTX_3080_PEAK_BW * 100) << "\n";
    std::cout << std::left << std::setw(16) << m_v2.kernel_name
              << std::right << std::setw(12) << std::fixed << std::setprecision(4) << m_v2.time_ms
              << std::setw(16) << std::fixed << std::setprecision(4) << m_v2.bandwidth_gbs
              << std::setw(14) << std::fixed << std::setprecision(2) << (m_v2.bandwidth_gbs / ProfilingMetrics::RTX_3080_PEAK_BW * 100) << "\n";

    std::cout << "\nResults written to: " << output_csv << "\n";
    std::cout << "=======================================================\n";

    return 0;
}