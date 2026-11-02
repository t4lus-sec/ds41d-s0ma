// s0ma change 32 (N4): programmatic dependent launch inside ds41d's captured graphs (DS41_PDL=1|2).
//
// Every kernel ds41d compiles starts with DS41_PDL_WAIT() (griddepcontrol.wait): it waits until the grids it
// depends on programmatically have completed and their writes are visible, and is a no-op otherwise. So an edge
// A -> B between two kernel nodes may become a programmatic edge whenever B is one of those kernels: B is launched
// before A has completed (DS41_PDL=1: once all of A's blocks have started; 2: once they have all exited), its
// blocks are resident and waiting when A ends, and B runs exactly what it ran before, after A's writes landed.
// What overlaps is B's launch latency and block scheduling. Edges into kernels ds41d does not compile (cuBLAS,
// cuBLASLt) and every edge touching a memcpy, memset or event node stay as captured.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
const std::unordered_set<std::string> &ours() {
  static const std::unordered_set<std::string> names = {
#include "pdl_kernels.inc"
  };
  return names;
}

// "void ds41::exl3_detail::decode_var<5120>(ds41::DeviceExpert const*, ...)" -> {"ds41", "exl3_detail"}, "decode_var"
bool waits_by_name(const char *mangled) {
  std::string s = mangled;
  int status = 0;
  if (char *d = abi::__cxa_demangle(mangled, nullptr, nullptr, &status)) {
    if (status == 0) s = d;
    std::free(d);
  }
  for (size_t at; (at = s.find("(anonymous namespace)")) != std::string::npos;) s.replace(at, 21, "@anon");
  if (!s.empty() && s.back() == ')') {   // drop the parameter list
    int depth = 0;
    for (size_t i = s.size(); i-- > 0;) {
      if (s[i] == ')') ++depth;
      else if (s[i] == '(' && --depth == 0) { s.resize(i); break; }
    }
  }
  if (!s.empty() && s.back() == '>') {   // drop the template arguments
    int depth = 0;
    for (size_t i = s.size(); i-- > 0;) {
      if (s[i] == '>') ++depth;
      else if (s[i] == '<' && --depth == 0) { s.resize(i); break; }
    }
  }
  if (const size_t sp = s.rfind(' '); sp != std::string::npos) s = s.substr(sp + 1);   // the return type
  std::vector<std::string> parts;
  for (size_t b = 0;;) {
    const size_t e = s.find("::", b);
    parts.push_back(s.substr(b, e == std::string::npos ? std::string::npos : e - b));
    if (e == std::string::npos) break;
    b = e + 2;
  }
  if (!ours().count(parts.back())) return false;
  static const std::unordered_set<std::string> foreign = {"cub", "thrust", "cutlass", "internal", "cublas", "cublasLt"};
  for (size_t i = 0; i + 1 < parts.size(); ++i)
    if (foreign.count(parts[i])) return false;
  // a global-scope name only when it is unmistakably ds41d's (its extern "C" attention kernels)
  if (parts.size() == 1) return parts[0].rfind("ds41_", 0) == 0 || parts[0] == "sparse_attn_kernel__kernel";
  return parts[0] == "ds41" || parts[0] == "ds41e" || parts[0] == "@anon";
}

bool kernel(cudaGraphNode_t node) {
  cudaGraphNodeType t;
  return cudaGraphNodeGetType(node, &t) == cudaSuccess && t == cudaGraphNodeTypeKernel;
}

bool waits(cudaGraphNode_t node) {
  static std::mutex m;
  static std::unordered_map<const void *, bool> cache;
  CUDA_KERNEL_NODE_PARAMS p{};
  if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &p) != CUDA_SUCCESS) return false;
  const void *key = p.func ? static_cast<const void *>(p.func) : static_cast<const void *>(p.kern);
  if (!key) return false;
  std::lock_guard<std::mutex> lock(m);
  if (auto it = cache.find(key); it != cache.end()) return it->second;
  const char *name = nullptr;
  const CUresult rc = p.func ? cuFuncGetName(&name, p.func) : cuKernelGetName(&name, p.kern);
  const bool ok = rc == CUDA_SUCCESS && name && waits_by_name(name);
  cache.emplace(key, ok);
  return ok;
}
}  // namespace

// The mode of DS41_PDL (0 off, 1 launch-completion edges, 2 programmatic edges); throws on another value.
extern "C" int ds41_pdl_mode() {
  static const int mode = [] {
    const char *v = std::getenv("DS41_PDL");
    const std::string s = v ? v : "";
    if (s.empty() || s == "0") return 0;
    if (s == "1") return 1;
    if (s == "2") return 2;
    return -1;
  }();
  return mode;
}

// Rewrites graph's eligible kernel->kernel edges in place; *edges gets their number. Returns a CUDA error code and
// names the failing step in *step. Captured graphs can already hold programmatic edges (cuBLAS launches some of its
// kernels with PDL), and a count-only query of such a graph is refused as lossy (cudaErrorLossyQuery): the edges are
// read with their data into arrays grown until they hold all of them.
extern "C" int ds41_pdl_rewrite(cudaGraph_t graph, int mode, size_t *edges, const char **step) {
  *edges = 0;
  *step = "reading the edges";
  std::vector<cudaGraphNode_t> from, to;
  std::vector<cudaGraphEdgeData> data;
  size_t n = 0;
  cudaError_t rc = cudaSuccess;
  for (size_t cap = 1024;; cap *= 4) {
    from.assign(cap, nullptr);
    to.assign(cap, nullptr);
    data.assign(cap, cudaGraphEdgeData{});
    n = cap;
    if ((rc = cudaGraphGetEdges(graph, from.data(), to.data(), data.data(), &n)) != cudaSuccess) return rc;
    if (n < cap) break;
    if (cap > (size_t(1) << 22)) return cudaErrorInvalidValue;
  }
  *step = "choosing the edges";
  std::vector<cudaGraphNode_t> f, t;
  std::vector<cudaGraphEdgeData> old, now;
  for (size_t i = 0; i < n; ++i) {
    if (data[i].type != cudaGraphDependencyTypeDefault || data[i].from_port || data[i].to_port) continue;
    if (!kernel(from[i]) || !kernel(to[i]) || !waits(to[i])) continue;
    cudaGraphEdgeData e{};
    e.type = cudaGraphDependencyTypeProgrammatic;
    e.from_port = mode == 1 ? cudaGraphKernelNodePortLaunchCompletion : cudaGraphKernelNodePortProgrammatic;
    f.push_back(from[i]);
    t.push_back(to[i]);
    old.push_back(data[i]);
    now.push_back(e);
  }
  if (f.empty()) return cudaSuccess;
  *step = "removing the captured edges";
  if ((rc = cudaGraphRemoveDependencies(graph, f.data(), t.data(), old.data(), f.size())) != cudaSuccess) return rc;
  *step = "adding the programmatic edges";
  if ((rc = cudaGraphAddDependencies(graph, f.data(), t.data(), now.data(), f.size())) != cudaSuccess) return rc;
  *edges = f.size();
  *step = "";
  return cudaSuccess;
}

// The kernel name of a graph node, for a refusal's message ("" if it is not a kernel node).
extern "C" const char *ds41_pdl_node_name(cudaGraphNode_t node) {
  if (!node || !kernel(node)) return "";
  CUDA_KERNEL_NODE_PARAMS p{};
  if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &p) != CUDA_SUCCESS) return "";
  const char *name = nullptr;
  if ((p.func ? cuFuncGetName(&name, p.func) : cuKernelGetName(&name, p.kern)) != CUDA_SUCCESS || !name) return "";
  return name;
}

// "kernel 120, memcpy 4, mem alloc 2, ..." for a refusal's message: which node types the graph holds
extern "C" const char *ds41_pdl_node_types(cudaGraph_t graph) {
  static thread_local std::string out;
  size_t n = 0;
  out.clear();
  if (cudaGraphGetNodes(graph, nullptr, &n) != cudaSuccess) return "?";
  std::vector<cudaGraphNode_t> nodes(n);
  if (n && cudaGraphGetNodes(graph, nodes.data(), &n) != cudaSuccess) return "?";
  std::map<int, size_t> count;
  for (auto node : nodes) {
    cudaGraphNodeType t;
    if (cudaGraphNodeGetType(node, &t) == cudaSuccess) ++count[int(t)];
  }
  static const char *names[] = {"kernel", "memcpy", "memset", "host", "child graph", "empty", "event wait",
                                "event record", "semaphore signal", "semaphore wait", "mem alloc", "mem free",
                                "batch memop", "conditional"};
  for (auto &[t, c] : count) {
    if (!out.empty()) out += ", ";
    out += (t >= 0 && t < int(sizeof names / sizeof *names) ? names[t] : "type " + std::to_string(t));
    out += " " + std::to_string(c);
  }
  return out.c_str();
}
