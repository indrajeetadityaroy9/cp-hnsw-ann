#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "index.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>

namespace py = pybind11;
using namespace cphnsw;

class PyIndexBase {
public:
    virtual ~PyIndexBase() = default;

    virtual void build(const float* vecs, size_t n) = 0;
    virtual void finalize() = 0;

    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;

    virtual std::vector<SearchResult>
    search_raw(const float* query, size_t k) const = 0;

    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

template <size_t D, size_t BitWidth>
class PyIndexWrapper : public PyIndexBase {
public:
    using IndexType = Index<D, BitWidth>;

    explicit PyIndexWrapper(size_t dim) {
        index_ = std::make_unique<IndexType>(dim);
    }

    void build(const float* vecs, size_t n) override {
        index_->build(vecs, n);
    }

    void finalize() override {
        index_->finalize();
    }

    size_t size() const override { return index_->size(); }
    size_t dim() const override { return index_->dim(); }

    std::vector<SearchResult>
    search_raw(const float* query, size_t k) const override {
        return index_->search(query, k);
    }

    void save(const std::string& path) const override {
        index_->save(path);
    }

    void load(const std::string& path) override {
        index_->load(path);
    }

private:
    std::unique_ptr<IndexType> index_;
};

template <size_t BitWidth>
static std::unique_ptr<PyIndexBase> create_index_with_bits(size_t dim) {
    size_t pd = next_power_of_two(dim);

#define CASE_DIM(DIM) \
    case DIM: return std::make_unique<PyIndexWrapper<DIM, BitWidth>>(dim);

    switch (pd) {
        CASE_DIM(16)
        CASE_DIM(32)
        CASE_DIM(64)
        CASE_DIM(128)
        CASE_DIM(256)
        CASE_DIM(512)
        CASE_DIM(1024)
        CASE_DIM(2048)
        default: __builtin_unreachable();
    }

#undef CASE_DIM
}

static std::unique_ptr<PyIndexBase> create_index(size_t dim, size_t bits) {
    switch (bits) {
        case 1: return create_index_with_bits<1>(dim);
        case 2: return create_index_with_bits<2>(dim);
        case 4: return create_index_with_bits<4>(dim);
        default: __builtin_unreachable();
    }
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Calibration-Parameterless HNSW (CP-HNSW)";

    py::class_<PyIndexBase>(m, "CPIndex")
        .def(py::init([](size_t dim, size_t bits) {
                return create_index(dim, bits);
            }),
            py::arg("dim"),
            py::arg("bits") = 1)

        .def("build", [](PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
            auto vbuf = vectors.request();
            const float* vec_ptr = static_cast<const float*>(vbuf.ptr);
            size_t n = static_cast<size_t>(vbuf.shape[0]);

            py::gil_scoped_release release;
            self.build(vec_ptr, n);
        },
        py::arg("vectors"))

        .def("finalize", [](PyIndexBase& self) {
            py::gil_scoped_release release;
            self.finalize();
        })

        .def("search", [](const PyIndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k) {
            auto buf = query.request();
            const float* ptr = static_cast<const float*>(buf.ptr);

            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_raw(ptr, k);
            }

            const size_t n = results.size();
            py::array_t<int64_t> ids(n);
            py::array_t<float> distances(n);
            auto* ids_ptr = ids.mutable_data();
            auto* dist_ptr = distances.mutable_data();
            for (size_t i = 0; i < n; ++i) {
                ids_ptr[i] = results[i].id;
                dist_ptr[i] = results[i].distance;
            }
            return std::make_pair(ids, distances);
        },
        py::arg("query"),
        py::arg("k") = 10)

        .def("search_batch", [](const PyIndexBase& self,
                                py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                size_t k) {
            auto buf = queries.request();
            const size_t n = static_cast<size_t>(buf.shape[0]);
            const float* ptr = static_cast<const float*>(buf.ptr);
            const size_t dim = self.dim();

            py::array_t<int64_t> ids({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
            py::array_t<float> distances({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
            auto* ids_ptr = ids.mutable_data();
            auto* dist_ptr = distances.mutable_data();

            {
                py::gil_scoped_release release;
                const int actual_threads = omp_get_max_threads();
#pragma omp parallel for schedule(guided) num_threads(actual_threads)
                for (size_t i = 0; i < n; ++i) {
                    auto results = self.search_raw(ptr + i * dim, k);
                    size_t j = 0;
                    for (; j < k && j < results.size(); ++j) {
                        ids_ptr[i * k + j] = static_cast<int64_t>(results[j].id);
                        dist_ptr[i * k + j] = results[j].distance;
                    }
                    for (; j < k; ++j) {
                        ids_ptr[i * k + j] = -1;
                        dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                    }
                }
            }

            return std::make_pair(ids, distances);
        },
        py::arg("queries"),
        py::arg("k") = 10)

        .def("save", [](const PyIndexBase& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
        },
        py::arg("path"))

        .def("load", [](PyIndexBase& self, const std::string& path) {
            py::gil_scoped_release release;
            self.load(path);
        },
        py::arg("path"))

        .def_property_readonly("size", &PyIndexBase::size)
        .def_property_readonly("dim", &PyIndexBase::dim);
}
