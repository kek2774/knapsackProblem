#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

namespace py = pybind11;

// Типы
struct Item {
    int weight;
    int price;
};

struct SolverResult {
    std::vector<Item> picked_items;
    int total_value;
    double elapsed_sec;
};

// Генерация предметов
enum class Distribution { EQUAL = 1, REVERSE_DOWN = 2, REVERSE_UP = 3, CHAOTIC = 4 };

std::vector<Item> generate_items(int count, Distribution dist,
                                  int w_min = 1, int w_max = 50, unsigned seed = 0) {
    std::mt19937 rng(seed ? seed : std::random_device{}());
    std::uniform_int_distribution<int> w_dist(w_min, w_max);
    std::uniform_int_distribution<int> p_dist(w_min * 100, w_max * 100);

    std::vector<Item> items;
    items.reserve(count);

    for (int i = 0; i < count; ++i) {
        int w = w_dist(rng);
        int p = 0;
        switch (dist) {
            case Distribution::EQUAL:
                p = w * 100;
                break;
            case Distribution::REVERSE_UP:
                p = w * w;
                break;
            case Distribution::REVERSE_DOWN:
                p = static_cast<int>(std::ceil(100.0 / std::sqrt(static_cast<double>(w))));
                break;
            case Distribution::CHAOTIC:
                p = p_dist(rng) * 100;
                break;
        }
        items.push_back({w, p});
    }
    return items;
}

// Жадный алгоритм (с коэффициентами в степени)
SolverResult greedy_solve(const std::vector<Item>& items, int capacity,
                          double alpha = 1.0, double beta = 1.0) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<size_t> idx(items.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    
    std::vector<double> key(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        key[i] = alpha * std::log(static_cast<double>(items[i].price))
               - beta  * std::log(static_cast<double>(items[i].weight));
    }

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return key[a] > key[b];
    });

    std::vector<Item> picked;
    int remaining = capacity;
    int total = 0;

    for (size_t i : idx) {
        if (items[i].weight <= remaining) {
            picked.push_back(items[i]);
            remaining -= items[i].weight;
            total += items[i].price;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    return {picked, total, elapsed};
}

// ДП 
SolverResult dynamic_solve(const std::vector<Item>& items, int capacity) {
    auto start = std::chrono::high_resolution_clock::now();

    int N = static_cast<int>(items.size());
    int W = capacity;

    std::vector<std::vector<int>> DP(N + 1, std::vector<int>(W + 1, 0));

    for (int i = 1; i <= N; ++i) {
        int wi = items[i - 1].weight;
        int vi = items[i - 1].price;
        for (int w = 1; w <= W; ++w) {
            if (wi <= w) {
                DP[i][w] = std::max(DP[i - 1][w], vi + DP[i - 1][w - wi]);
            } else {
                DP[i][w] = DP[i - 1][w];
            }
        }
    }

    std::vector<Item> picked;
    int i = N, w = W;
    while (i > 0 && w > 0) {
        if (DP[i][w] != DP[i - 1][w]) {
            picked.push_back(items[i - 1]);
            w -= items[i - 1].weight;
        }
        --i;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    return {picked, DP[N][W], elapsed};
}

// Полный прогон 
struct RunRow {
    int items_count;
    double time_greedy;
    double time_ds;
    int greedy_value;
    int ds_value;
    int greedy_count;
    int ds_count;
    double error_pct;
    long int sum_item_weights;
    long int sum_dp_weights;
    std::vector<int> greedy_picked_weights;
    std::vector<int> greedy_picked_prices;
    std::vector<int> dp_picked_weights;
    std::vector<int> dp_picked_prices;
};

long int sum_weights(const std::vector<Item>& items) {
    long int summa = 0;
    for (std::size_t i = 0; i < items.size(); i++) {
        summa += items[i].weight;
    }
    return summa;
}

std::vector<RunRow> run_benchmark(Distribution dist, const std::vector<int>& counts,
                                   int capacity, int w_min, int w_max, unsigned seed,
                                   double alpha = 1.0, double beta = 1.0) {
    std::vector<RunRow> rows;
    rows.reserve(counts.size());

    for (int n : counts) {
        auto items = generate_items(n, dist, w_min, w_max, seed);

        auto gr = greedy_solve(items, capacity, alpha, beta);
        auto dp = dynamic_solve(items, capacity);

        double err = dp.total_value > 0
            ? (dp.total_value - gr.total_value) / static_cast<double>(dp.total_value) * 100.0
            : 0.0;

        std::vector<int> gpw, gpp, dpw, dpp;
        for (const auto& it : gr.picked_items) {
            gpw.push_back(it.weight);
            gpp.push_back(it.price);
        }
        for (const auto& it : dp.picked_items) {
            dpw.push_back(it.weight);
            dpp.push_back(it.price);
        }

        rows.push_back({
            static_cast<int>(items.size()),
            gr.elapsed_sec,
            dp.elapsed_sec,
            gr.total_value,
            dp.total_value,
            static_cast<int>(gr.picked_items.size()),
            static_cast<int>(dp.picked_items.size()),
            err,
            sum_weights(items),
            sum_weights(dp.picked_items),
            gpw, gpp, dpw, dpp
        });
    }
    return rows;
}

struct ItemArrays {
    std::vector<int> weights;
    std::vector<int> prices;
};

ItemArrays get_item_arrays(Distribution dist, int count, int w_min, int w_max, unsigned seed) {
    auto items = generate_items(count, dist, w_min, w_max, seed);
    ItemArrays result;
    result.weights.reserve(count);
    result.prices.reserve(count);
    for (const auto& item : items) {
        result.weights.push_back(item.weight);
        result.prices.push_back(item.price);
    }
    return result;
}

// pybind11 
PYBIND11_MODULE(knapsack_cpp, m) {
    m.doc() = "C++ backend for the knapsack benchmark";

    py::enum_<Distribution>(m, "Distribution")
        .value("EQUAL",        Distribution::EQUAL)
        .value("REVERSE_DOWN", Distribution::REVERSE_DOWN)
        .value("REVERSE_UP",   Distribution::REVERSE_UP)
        .value("CHAOTIC",      Distribution::CHAOTIC);

    py::class_<Item>(m, "Item")
        .def(py::init<int, int>())
        .def_readwrite("weight", &Item::weight)
        .def_readwrite("price",  &Item::price);

    py::class_<SolverResult>(m, "SolverResult")
        .def_readwrite("picked_items", &SolverResult::picked_items)
        .def_readwrite("total_value",  &SolverResult::total_value)
        .def_readwrite("elapsed_sec",  &SolverResult::elapsed_sec);

    py::class_<RunRow>(m, "RunRow")
        .def_readwrite("items_count",   &RunRow::items_count)
        .def_readwrite("time_greedy",   &RunRow::time_greedy)
        .def_readwrite("time_ds",       &RunRow::time_ds)
        .def_readwrite("greedy_value",  &RunRow::greedy_value)
        .def_readwrite("ds_value",      &RunRow::ds_value)
        .def_readwrite("greedy_count",  &RunRow::greedy_count)
        .def_readwrite("ds_count",      &RunRow::ds_count)
        .def_readwrite("error_pct",     &RunRow::error_pct)
        .def_readwrite("sum_item_weights", &RunRow::sum_item_weights)
        .def_readwrite("sum_dp_weights",   &RunRow::sum_dp_weights)
        .def_readwrite("greedy_picked_weights", &RunRow::greedy_picked_weights)
        .def_readwrite("greedy_picked_prices",  &RunRow::greedy_picked_prices)
        .def_readwrite("dp_picked_weights",     &RunRow::dp_picked_weights)
        .def_readwrite("dp_picked_prices",      &RunRow::dp_picked_prices);

    py::class_<ItemArrays>(m, "ItemArrays")
        .def_readwrite("weights", &ItemArrays::weights)
        .def_readwrite("prices",  &ItemArrays::prices);

    m.def("generate_items", &generate_items,
          py::arg("count"), py::arg("dist"),
          py::arg("w_min") = 1, py::arg("w_max") = 50, py::arg("seed") = 0);

    m.def("greedy_solve", &greedy_solve,
          py::arg("items"), py::arg("capacity"),
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0);

    m.def("dynamic_solve", &dynamic_solve,
          py::arg("items"), py::arg("capacity"));

    m.def("run_benchmark", &run_benchmark,
          py::arg("dist"), py::arg("counts"),
          py::arg("capacity") = 150,
          py::arg("w_min") = 1, py::arg("w_max") = 50,
          py::arg("seed") = 0,
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
          "Run full benchmark for one distribution across all item counts");

    m.def("get_item_arrays", &get_item_arrays,
          py::arg("dist"), py::arg("count"),
          py::arg("w_min") = 1, py::arg("w_max") = 50, py::arg("seed") = 0,
          "Return weight and price arrays for histogram computation");
}