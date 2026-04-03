#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "index.hpp"

#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

namespace py = pybind11;
using namespace evtq;

static std::unique_ptr<IndexBase> create_index(size_t dim) {
    size_t pd = next_power_of_two(dim);

#define CASE_DIM(DIM) \
    case DIM: return std::make_unique<Index<DIM>>(dim);

    switch (pd) {
        CASE_DIM(16)
        CASE_DIM(32)
        CASE_DIM(64)
        CASE_DIM(128)
        CASE_DIM(256)
        CASE_DIM(512)
        CASE_DIM(1024)
        CASE_DIM(2048)
        default: std::unreachable();
    }

#undef CASE_DIM
}

PYBIND11_MODULE(evtq, m) {
    py::class_<IndexBase>(m, "EVTQIndex")
        .def(py::init([](size_t dim) {
                return create_index(dim);
            }),
            py::arg("dim"))

        .def("build", [](IndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
            auto vbuf = vectors.request();
            auto vec_ptr = static_cast<const float*>(vbuf.ptr);
            auto n = static_cast<size_t>(vbuf.shape[0]);
            auto total = static_cast<size_t>(vbuf.size);

            py::gil_scoped_release release;
            self.build(std::span<const float>{vec_ptr, total}, n);
        },
        py::arg("vectors"))

        .def("finalize", [](IndexBase& self) {
            py::gil_scoped_release release;
            self.finalize();
        })

        .def("search", [](const IndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k) {
            auto buf = query.request();
            auto ptr = static_cast<const float*>(buf.ptr);
            auto len = static_cast<size_t>(buf.size);

            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search(std::span<const float>{ptr, len}, k);
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

        .def("search_exhaustive", [](const IndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k) {
            auto buf = query.request();
            auto ptr = static_cast<const float*>(buf.ptr);
            auto len = static_cast<size_t>(buf.size);

            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_exhaustive(std::span<const float>{ptr, len}, k);
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

        .def("search_graph_ceiling", [](const IndexBase& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          size_t k) {
            auto buf = query.request();
            auto ptr = static_cast<const float*>(buf.ptr);
            auto len = static_cast<size_t>(buf.size);

            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_graph_ceiling(std::span<const float>{ptr, len}, k);
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

        .def("search_batch", [](const IndexBase& self,
                                py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                size_t k) {
            auto buf = queries.request();
            auto n = static_cast<size_t>(buf.shape[0]);
            auto ptr = static_cast<const float*>(buf.ptr);
            auto dim = self.dim();

            py::array_t<int64_t> ids({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
            py::array_t<float> distances({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
            auto* ids_ptr = ids.mutable_data();
            auto* dist_ptr = distances.mutable_data();

            {
                py::gil_scoped_release release;
                const int actual_threads = omp_get_max_threads();
#pragma omp parallel for schedule(guided) num_threads(actual_threads)
                for (size_t i = 0; i < n; ++i) {
                    auto results = self.search(
                        std::span<const float>{ptr + i * dim, dim}, k);
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

        .def("save", [](const IndexBase& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
        },
        py::arg("path"))

        .def("load", [](IndexBase& self, const std::string& path) {
            py::gil_scoped_release release;
            self.load(path);
        },
        py::arg("path"))

        .def_property_readonly("size", &IndexBase::size)
        .def_property_readonly("dim", &IndexBase::dim);
}
