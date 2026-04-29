// Combined device counter sampling + kernel dispatch tracing tool.
// Based on rocm-examples device_profiling/client.cpp (counter sampling)
// with kernel dispatch tracing added from api_buffered_tracing/client.cpp.
//
// Loaded via LD_PRELOAD. Outputs kernel_traces.csv and counter_samples.csv.
//
// Env var configuration:
//   CHOPPER_COUNTERS     - comma-separated counter names (default: SQ_WAVES)
//   CHOPPER_SAMPLE_MS    - sampling interval in ms (default: 1)
//   CHOPPER_COUNTER_OUTPUT - counter CSV path (default: counter_samples.csv)
//   CHOPPER_TRACE_OUTPUT   - trace CSV path (default: kernel_traces.csv)

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// Error checking macro (matches rocprofiler_get_status_string signature on ROCm 7.x)
#define ROCPROFILER_CALL(result, msg)                                                    \
    {                                                                                    \
        rocprofiler_status_t CHECKSTATUS = result;                                       \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                    \
        {                                                                                \
            const char* status_msg = rocprofiler_get_status_string(CHECKSTATUS);         \
            std::cerr << "[tool] " << msg << " failed with status " << CHECKSTATUS       \
                      << " (" << (status_msg ? status_msg : "unknown") << ")\n";         \
            std::abort();                                                                \
        }                                                                                \
    }

namespace
{

//
// Kernel symbol tracking (from api_buffered_tracing example)
//
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

kernel_symbol_map_t client_kernels   = {};
std::mutex          client_kernels_mutex;

//
// Kernel trace records collected in buffer callback
//
struct kernel_record_t
{
    std::string kernel_name;
    uint64_t    start_ns;
    uint64_t    end_ns;
    uint64_t    agent_id;
    uint64_t    queue_id;
    uint64_t    correlation_id;
};
std::vector<kernel_record_t> g_kernel_records;
std::mutex                   g_kernel_records_mutex;

//
// Counter dimension metadata (per counter_id)
//
struct dim_info_t
{
    std::string                        name;
    rocprofiler_counter_dimension_id_t id;
    size_t                             size;
};

struct counter_dim_info_t
{
    std::string             name;
    std::vector<dim_info_t> dims;
};
std::unordered_map<uint64_t, counter_dim_info_t> g_counter_dims;
std::vector<std::string>                         g_all_dim_names;

//
// Counter sample records
//
struct counter_sample_t
{
    uint64_t                                timestamp_ns;
    std::string                             counter_name;
    double                                  value;
    uint64_t                                agent_id;
    std::unordered_map<std::string, size_t> dim_positions;
};
std::vector<counter_sample_t> g_counter_samples;
std::mutex                    g_counter_samples_mutex;

//
// Shared state
//
rocprofiler_context_id_t g_ctx{0};
rocprofiler_buffer_id_t  g_trace_buffer{};
rocprofiler_buffer_id_t  g_counter_buffer{};
std::atomic<bool>        g_exit{false};
int                      g_sample_interval_ms = 1;

std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> g_profile_cache;

//
// Code object callback -- from api_buffered_tracing example.
// Tracks kernel symbols for name resolution and flushes the trace buffer
// before code objects unload so kernel name pointers remain valid.
//
void tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t*              user_data,
                               void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
       && record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(g_trace_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                (void)flush_status;
            }
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
            && record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            std::lock_guard<std::mutex> lk(client_kernels_mutex);
            client_kernels.emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // do not erase just in case a buffer callback needs this
            // client_kernels.erase(data->kernel_id);
        }
    }

    (void)user_data;
    (void)callback_data;
}

//
// Buffer callback -- handles both tracing records and counter records.
// Kernel dispatch handling adapted from api_buffered_tracing example.
// Counter record handling adapted from device_profiling example.
//
void tool_buffer_callback(rocprofiler_context_id_t,
                          rocprofiler_buffer_id_t,
                          rocprofiler_record_header_t** headers,
                          size_t                        num_headers,
                          void*,
                          uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING
           && header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            // Kernel dispatch record -- from api_buffered_tracing example
            auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                header->payload);

            auto kernel_id   = record->dispatch_info.kernel_id;
            std::string kernel_name = "??";
            {
                std::lock_guard<std::mutex> lk(client_kernels_mutex);
                auto it = client_kernels.find(kernel_id);
                if(it != client_kernels.end())
                    kernel_name = it->second.kernel_name;
            }

            std::lock_guard<std::mutex> lk(g_kernel_records_mutex);
            g_kernel_records.push_back({std::move(kernel_name),
                                        record->start_timestamp,
                                        record->end_timestamp,
                                        record->dispatch_info.agent_id.handle,
                                        record->dispatch_info.queue_id.handle,
                                        record->correlation_id.internal});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS
                && header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            // skip dispatch headers (from device_profiling example)
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS
                && header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Counter record -- from device_profiling example, extended with dimension decoding
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);

            rocprofiler_counter_id_t counter_id = {.handle = 0};
            rocprofiler_query_record_counter_id(record->id, &counter_id);

            std::string counter_name = "??";
            std::unordered_map<std::string, size_t> positions;

            auto it = g_counter_dims.find(counter_id.handle);
            if(it != g_counter_dims.end())
            {
                counter_name = it->second.name;
                for(const auto& dim : it->second.dims)
                {
                    size_t pos = 0;
                    rocprofiler_query_record_dimension_position(record->id, dim.id, &pos);
                    positions[dim.name] = pos;
                }
            }

            std::lock_guard<std::mutex> lk(g_counter_samples_mutex);
            g_counter_samples.push_back({record->user_data.value,
                                         counter_name,
                                         record->counter_value,
                                         record->agent_id.handle,
                                         std::move(positions)});
        }
    }
}

//
// Device counting: set_profile callback -- from device_profiling example
//
void set_profile(rocprofiler_context_id_t               context_id,
                 rocprofiler_agent_id_t                 agent,
                 rocprofiler_device_counting_agent_cb_t set_config,
                 void*)
{
    auto search_cache = [&]()
    {
        if(auto pos = g_profile_cache.find(agent.handle); pos != g_profile_cache.end())
        {
            set_config(context_id, pos->second);
            return true;
        }
        return false;
    };

    if(!search_cache())
    {
        std::cerr << "No profile for agent found in cache\n";
        exit(-1);
    }
}

//
// Build counter profile for an agent -- from device_profiling example,
// extended to read CHOPPER_COUNTERS env var and discover dimensions.
//
rocprofiler_counter_config_id_t build_profile_for_agent(rocprofiler_agent_id_t agent)
{
    std::set<std::string> counters_to_collect = {"SQ_WAVES"};

    if(auto* env = std::getenv("CHOPPER_COUNTERS"); env)
    {
        std::istringstream ss(env);
        std::string        token;
        counters_to_collect.clear();
        while(std::getline(ss, token, ','))
        {
            if(!token.empty()) counters_to_collect.insert(token);
        }
    }

    std::vector<rocprofiler_counter_id_t> gpu_counters;
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data)
                         {
                             std::vector<rocprofiler_counter_id_t>* vec
                                 = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

    std::vector<rocprofiler_counter_id_t> collect_counters;
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(rocprofiler_query_counter_info(counter,
                                                        ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                        static_cast<void*>(&info)),
                         "Could not query info for counter");
        if(counters_to_collect.count(std::string(info.name)) > 0)
        {
            std::clog << "[tool] Collecting counter: " << info.name << "\n";
            collect_counters.push_back(counter);
        }
    }

    // Discover dimensions per counter using v1 info
    for(auto& counter : collect_counters)
    {
        rocprofiler_counter_info_v1_t info_v1;
        info_v1.size = sizeof(rocprofiler_counter_info_v1_t);
        auto status = rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info_v1));
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            counter_dim_info_t cdi;
            cdi.name = info_v1.name;
            for(uint64_t d = 0; d < info_v1.dimensions_count; d++)
            {
                std::string dname = info_v1.dimensions[d]->name;
                cdi.dims.push_back({dname,
                                    info_v1.dimensions[d]->id,
                                    info_v1.dimensions[d]->instance_size});
                bool found = false;
                for(const auto& existing : g_all_dim_names)
                    if(existing == dname) { found = true; break; }
                if(!found)
                    g_all_dim_names.push_back(dname);
            }
            g_counter_dims[counter.handle] = std::move(cdi);
        }
    }

    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(agent,
                                                       collect_counters.data(),
                                                       collect_counters.size(),
                                                       &profile),
                     "Could not construct profile cfg");

    return profile;
}

//
// tool_init -- combines setup from both examples into one context.
//
int tool_init(rocprofiler_client_finalize_t, void*)
{
    if(auto* env = std::getenv("CHOPPER_SAMPLE_MS"); env)
        g_sample_interval_ms = std::atoi(env);

    std::clog << "[tool] Initializing with sample interval " << g_sample_interval_ms << " ms\n";

    ROCPROFILER_CALL(rocprofiler_create_context(&g_ctx), "context creation failed");

    // Code object callback for kernel name resolution (from api_buffered_tracing)
    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_LOAD,
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(g_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       code_object_ops.data(),
                                                       code_object_ops.size(),
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

    // Trace buffer for kernel dispatch records
    ROCPROFILER_CALL(rocprofiler_create_buffer(g_ctx,
                                               16384,
                                               16384 - 2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_buffer_callback,
                                               nullptr,
                                               &g_trace_buffer),
                     "trace buffer creation");

    // Configure kernel dispatch tracing (from api_buffered_tracing)
    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(g_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                     nullptr,
                                                     0,
                                                     g_trace_buffer),
        "buffer tracing service for kernel dispatch configure");

    // Counter buffer for device counting records
    ROCPROFILER_CALL(rocprofiler_create_buffer(g_ctx,
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_buffer_callback,
                                               nullptr,
                                               &g_counter_buffer),
                     "counter buffer creation");

    // Discover GPU agents and build counter profiles (from device_profiling)
    std::vector<rocprofiler_agent_v0_t>     agents;
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata)
    {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
        {
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        }
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            agents_v->emplace_back(*static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]));
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");

    // Select the GPU agent matching LOCAL_RANK (for multi-rank torchrun).
    // If LOCAL_RANK is not set, use the first GPU agent.
    int target_gpu_index = 0;
    if(auto* lr = std::getenv("LOCAL_RANK"); lr)
        target_gpu_index = std::atoi(lr);

    rocprofiler_agent_id_t target_agent = {.handle = 0};
    int gpu_index = 0;
    for(const auto& agent : agents)
    {
        if(agent.type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            if(gpu_index == target_gpu_index)
            {
                g_profile_cache.emplace(agent.id.handle, build_profile_for_agent(agent.id));
                target_agent = agent.id;
                std::clog << "[tool] rank " << target_gpu_index
                          << " using GPU agent: " << agent.id.handle << "\n";
                break;
            }
            gpu_index++;
        }
    }

    if(target_agent.handle == 0)
    {
        std::cerr << "[tool] No GPU agents found\n";
        return 1;
    }

    // Configure device counting service (from device_profiling)
    ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(g_ctx,
                                                                   g_counter_buffer,
                                                                   target_agent,
                                                                   set_profile,
                                                                   nullptr),
                     "Could not setup device counting service");

    // Create and assign callback thread for both buffers
    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(g_trace_buffer, client_thread),
                     "failed to assign thread for trace buffer");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(g_counter_buffer, client_thread),
                     "failed to assign thread for counter buffer");

    // Sampling thread.
    std::thread(
        [=]()
        {
            rocprofiler_start_context(g_ctx);
            std::clog << "[tool] Context started, sampling active\n";

            uint64_t count = 0;
            while(g_exit.load() == false)
            {
                rocprofiler_timestamp_t ts = 0;
                rocprofiler_get_timestamp(&ts);

                rocprofiler_sample_device_counting_service(g_ctx,
                                                           {.value = ts},
                                                           ROCPROFILER_COUNTER_FLAG_NONE,
                                                           nullptr,
                                                           nullptr);
                count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(g_sample_interval_ms));
            }
            g_exit.store(false);
            std::clog << "[tool] Sampling thread exiting after " << count << " samples\n";
        })
        .detach();

    std::clog << "[tool] Initialization complete\n";
    return 0;
}

//
// tool_fini -- stop sampling, flush buffers, write CSVs.
//
void tool_fini(void*)
{
    std::clog << "[tool] Finalizing...\n";

    g_exit.store(true);
    while(g_exit.load() == true)
    {};

    rocprofiler_stop_context(g_ctx);
    ROCPROFILER_CALL(rocprofiler_flush_buffer(g_trace_buffer), "trace buffer flush");
    ROCPROFILER_CALL(rocprofiler_flush_buffer(g_counter_buffer), "counter buffer flush");

    // Determine rank suffix for multi-process runs (torchrun sets LOCAL_RANK)
    std::string rank_suffix;
    if(auto* lr = std::getenv("LOCAL_RANK"); lr)
        rank_suffix = std::string("_rank") + lr;

    // Write kernel traces CSV
    {
        std::string path = "kernel_traces" + rank_suffix + ".csv";
        if(auto* env = std::getenv("CHOPPER_TRACE_OUTPUT"); env)
        {
            path = env;
            if(!rank_suffix.empty())
            {
                auto dot = path.rfind('.');
                if(dot != std::string::npos)
                    path = path.substr(0, dot) + rank_suffix + path.substr(dot);
                else
                    path += rank_suffix;
            }
        }

        std::ofstream out(path);
        out << "kernel_name,start_ns,end_ns,duration_ns,agent_id,queue_id,correlation_id\n";
        std::lock_guard<std::mutex> lk(g_kernel_records_mutex);
        for(const auto& r : g_kernel_records)
        {
            out << "\"" << r.kernel_name << "\","
                << r.start_ns << ","
                << r.end_ns << ","
                << (r.end_ns - r.start_ns) << ","
                << r.agent_id << ","
                << r.queue_id << ","
                << r.correlation_id << "\n";
        }
        std::clog << "[tool] Wrote " << g_kernel_records.size()
                  << " kernel records to " << path << "\n";
    }

    // Write counter samples CSV
    {
        std::string path = "counter_samples" + rank_suffix + ".csv";
        if(auto* env = std::getenv("CHOPPER_COUNTER_OUTPUT"); env)
        {
            path = env;
            if(!rank_suffix.empty())
            {
                auto dot = path.rfind('.');
                if(dot != std::string::npos)
                    path = path.substr(0, dot) + rank_suffix + path.substr(dot);
                else
                    path += rank_suffix;
            }
        }

        std::ofstream out(path);
        out << "timestamp_ns,counter_name,counter_value,agent_id";
        for(const auto& d : g_all_dim_names)
            out << "," << d;
        out << "\n";
        std::lock_guard<std::mutex> lk(g_counter_samples_mutex);
        for(const auto& s : g_counter_samples)
        {
            out << s.timestamp_ns << ","
                << s.counter_name << ","
                << s.value << ","
                << s.agent_id;
            for(const auto& d : g_all_dim_names)
            {
                auto it = s.dim_positions.find(d);
                out << "," << (it != s.dim_positions.end() ? it->second : 0);
            }
            out << "\n";
        }
        std::clog << "[tool] Wrote " << g_counter_samples.size()
                  << " counter samples to " << path << "\n";
    }

    std::clog << "[tool] Done.\n";
}

} // namespace

//
// rocprofiler-sdk entry point -- from device_profiling example
//
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t    version,
                                                                      const char* runtime_version,
                                                                      uint32_t    priority,
                                                                      rocprofiler_client_id_t* id)
{
    id->name = "chopper_device_profiler";

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    static auto cfg
        = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                              &tool_init,
                                              &tool_fini,
                                              nullptr};

    return &cfg;
}
