from enum import Enum

import flet as ft
import flet_charts as fch
import knapsack_cpp  # C++ модуль
import matplotlib.pyplot as plt
import numpy as np

# Константы
cCapacity = 400
cSeed = 225  # Рандомный сид seed = 0 Интересные: 54,225
cStartRandWeight = 1
cEndRandWeight = 350


class Distributions(Enum):
    EQUAL = 1
    REVERSE_DEPENDENCY_DOWN = 2
    REVERSE_DEPENDENCY_UP = 3
    CHAOTIC = 4


# Python-enum -> C++ enum
_DIST_TO_CPP = {
    Distributions.EQUAL: knapsack_cpp.Distribution.EQUAL,
    Distributions.REVERSE_DEPENDENCY_DOWN: knapsack_cpp.Distribution.REVERSE_DOWN,
    Distributions.REVERSE_DEPENDENCY_UP: knapsack_cpp.Distribution.REVERSE_UP,
    Distributions.CHAOTIC: knapsack_cpp.Distribution.CHAOTIC,
}

DIST_LABELS = {
    Distributions.EQUAL: "Равномерное",
    Distributions.REVERSE_DEPENDENCY_DOWN: "Обратная зав. (↓)",
    Distributions.REVERSE_DEPENDENCY_UP: "Прямая зав. (↑)",
    Distributions.CHAOTIC: "Хаотичное",
}

DIST_COLORS = {
    Distributions.EQUAL: "#42A5F5",
    Distributions.REVERSE_DEPENDENCY_DOWN: "#FFA726",
    Distributions.REVERSE_DEPENDENCY_UP: "#66BB6A",
    Distributions.CHAOTIC: "#EF5350",
}

DIST_FLET_COLORS = {
    Distributions.EQUAL: "#42A5F5",
    Distributions.REVERSE_DEPENDENCY_DOWN: "#FFA726",
    Distributions.REVERSE_DEPENDENCY_UP: "#66BB6A",
    Distributions.CHAOTIC: "#EF5350",
}

DIST_ICONS = {
    Distributions.EQUAL: ft.Icons.LINEAR_SCALE,
    Distributions.REVERSE_DEPENDENCY_DOWN: ft.Icons.TRENDING_DOWN,
    Distributions.REVERSE_DEPENDENCY_UP: ft.Icons.TRENDING_UP,
    Distributions.CHAOTIC: ft.Icons.SHUFFLE,
}

OPTIMIZED_COEFFS = {
    Distributions.EQUAL: (2.4091, 0.6467),
    Distributions.REVERSE_DEPENDENCY_DOWN: (0.3337, 3.1651),
    Distributions.REVERSE_DEPENDENCY_UP: (1.5327, 3.0437),
    Distributions.CHAOTIC: (4.6225, 2.9507),
}


# Подготовка данных - вызов C++ бенчмарка
items_counts = [10**i for i in range(0, 8)]
all_results = {}

for dist in Distributions:
    cpp_dist = _DIST_TO_CPP[dist]
    alpha, beta = OPTIMIZED_COEFFS[dist]
    rows = knapsack_cpp.run_benchmark(
        dist=cpp_dist,
        counts=items_counts,
        capacity=cCapacity,
        w_min=cStartRandWeight,
        w_max=cEndRandWeight,
        seed=cSeed,
        alpha=alpha,
        beta=beta,
    )

    all_results[dist] = {
        "time_greedy": [r.time_greedy for r in rows],
        "time_ds": [r.time_ds for r in rows],
        "greedy_counts": [r.greedy_count for r in rows],
        "ds_counts": [r.ds_count for r in rows],
        "greedy_values": [r.greedy_value for r in rows],
        "ds_values": [r.ds_value for r in rows],
        "errors": [r.error_pct for r in rows],
        "items_counts": [r.items_count for r in rows],
        "sum_item_weights": [r.sum_item_weights for r in rows],
        "sum_dp_weights": [r.sum_dp_weights for r in rows],
        "greedy_picked_weights": [np.array(r.greedy_picked_weights) for r in rows],
        "greedy_picked_prices": [np.array(r.greedy_picked_prices) for r in rows],
        "dp_picked_weights": [np.array(r.dp_picked_weights) for r in rows],
        "dp_picked_prices": [np.array(r.dp_picked_prices) for r in rows],
    }

    # Предрасчёт массивов весов/ценностей для гистограмм
    item_arrays = {}
    for dist in Distributions:
        cpp_dist = _DIST_TO_CPP[dist]
        item_arrays[dist] = {}
        for i, n in enumerate(items_counts):
            arrays = knapsack_cpp.get_item_arrays(
                cpp_dist, n, cStartRandWeight, cEndRandWeight, 0
            )
            item_arrays[dist][i] = {
                "weights": np.array(arrays.weights),
                "prices": np.array(arrays.prices),
            }

# Для сравнения с базовым жадным алгоритмом
basic_greedy_errors = {}
basic_greedy_values = {}
for dist in Distributions:
    cpp_dist = _DIST_TO_CPP[dist]
    rows = knapsack_cpp.run_benchmark(
        dist=cpp_dist,
        counts=items_counts,
        capacity=cCapacity,
        w_min=cStartRandWeight,
        w_max=cEndRandWeight,
        seed=cSeed,
        alpha=1.0,
        beta=1.0,
    )
    basic_greedy_errors[dist] = [r.error_pct for r in rows]
    basic_greedy_values[dist] = [r.greedy_value for r in rows]

# Matplotlib стили
MPL_STYLE = {
    "axes.facecolor": "#1E1E2E",
    "figure.facecolor": "#1E1E2E",
    "axes.edgecolor": "#45475A",
    "axes.labelcolor": "#CDD6F4",
    "xtick.color": "#BAC2DE",
    "ytick.color": "#BAC2DE",
    "text.color": "#CDD6F4",
    "grid.color": "#45475A",
    "grid.alpha": 0.4,
    "legend.facecolor": "#313244",
    "legend.edgecolor": "#45475A",
    "legend.labelcolor": "#CDD6F4",
    "font.size": 11,
}

MPL_STYLE_LIGHT = {
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#F8F9FA",
    "axes.edgecolor": "#DEE2E6",
    "axes.labelcolor": "#212529",
    "xtick.color": "#495057",
    "ytick.color": "#495057",
    "text.color": "#212529",
    "grid.color": "#DEE2E6",
    "grid.alpha": 0.6,
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": "#DEE2E6",
    "legend.labelcolor": "#212529",
    "font.size": 11,
}


def apply_mpl_style(dark=True):
    style = MPL_STYLE if dark else MPL_STYLE_LIGHT
    for k, v in style.items():
        plt.rcParams[k] = v


apply_mpl_style(dark=True)


# Вспомогательные функции перерисовки


def _draw_empty(ax):
    ax.clear()
    ax.text(
        0.5,
        0.5,
        "Выберите хотя бы одно распределение",
        ha="center",
        va="center",
        fontsize=13,
        color="#6C7086",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _style_log_ax(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.8)


def _style_bar_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def redraw_line(ax, fig, active, data_key, xlabel, ylabel, title, marker="o"):
    ax.clear()
    if not active:
        _draw_empty(ax)
    else:
        ax.set_xscale("log")
        for dist in active:
            r = all_results[dist]
            ax.plot(
                items_counts,
                r[data_key],
                marker=marker,
                linewidth=2,
                markersize=6,
                label=DIST_LABELS[dist],
                color=DIST_COLORS[dist],
            )
        _style_log_ax(ax, xlabel, ylabel, title)
    fig.tight_layout()
    fig.canvas.draw_idle()


def redraw_bar_single_n(ax, fig, active, data_key, n_idx, ylabel, title):
    ax.clear()
    if not active:
        _draw_empty(ax)
    else:
        x = np.arange(len(active))
        colors = [DIST_COLORS[d] for d in active]
        values = [all_results[d][data_key][n_idx] for d in active]
        labels = [DIST_LABELS[d] for d in active]
        ax.bar(x, values, 0.5, color=colors, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        n_val = items_counts[n_idx]
        ax.set_title(
            f"{title}  (N = {n_val:,})".replace(",", " "),
            fontweight="bold",
            fontsize=13,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.canvas.draw_idle()


def redraw_compare_pair(
    axes, fig, active, key_left, key_right, ylabel, title_left, title_right
):
    for ax in axes:
        ax.clear()
    if not active:
        for ax in axes:
            _draw_empty(ax)
    else:
        for dist in active:
            r = all_results[dist]
            lbl = DIST_LABELS[dist]
            clr = DIST_COLORS[dist]
            axes[0].plot(
                items_counts,
                r[key_left],
                marker="o",
                linewidth=2,
                markersize=6,
                label=lbl,
                color=clr,
            )
            axes[1].plot(
                items_counts,
                r[key_right],
                marker="s",
                linewidth=2,
                markersize=6,
                label=lbl,
                color=clr,
            )
        for ax, title in zip(axes, [title_left, title_right]):
            ax.set_xscale("log")
            ax.set_xlabel("N")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold", fontsize=13)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.canvas.draw_idle()


NUM_HIST_BINS = 10


def _compute_bins(arrays_list, mode):
    """Общие границы бинов по всем переданным массивам."""
    if mode == "weight":
        return np.linspace(cStartRandWeight, cEndRandWeight + 1, NUM_HIST_BINS + 1)
    all_vals = np.concatenate([a for a in arrays_list if a.size > 0])
    if all_vals.size == 0:
        return np.linspace(0, 1, NUM_HIST_BINS + 1)
    v_min, v_max = float(all_vals.min()), float(all_vals.max())
    if v_min == v_max:
        v_max = v_min + 1
    return np.linspace(v_min, v_max + 1, NUM_HIST_BINS + 1)


def redraw_histogram_single(ax, fig, active, n_idx, mode, source_key_prefix):
    """
    Гистограмма для одной страницы (ДП или Жадный).
    source_key_prefix: 'dp_picked' или 'greedy_picked'
    """
    ax.clear()
    wkey = f"{source_key_prefix}_weights"
    pkey = f"{source_key_prefix}_prices"
    data_key = wkey if mode == "weight" else pkey

    if not active:
        _draw_empty(ax)
        fig.tight_layout()
        fig.canvas.draw_idle()
        return

    arrays = [all_results[d][data_key][n_idx] for d in active]
    if all(a.size == 0 for a in arrays):
        ax.text(
            0.5,
            0.5,
            "Нет выбранных предметов",
            ha="center",
            va="center",
            fontsize=13,
            color="#6C7086",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw_idle()
        return

    bins = _compute_bins(arrays, mode)
    x = np.arange(NUM_HIST_BINS)
    total = len(active)
    width = 0.7 / max(total, 1)

    for idx, dist in enumerate(active):
        data = all_results[dist][data_key][n_idx]
        counts_arr, _ = np.histogram(data, bins=bins)
        offset = (idx - total / 2 + 0.5) * width
        ax.bar(
            x + offset,
            counts_arr,
            width,
            label=DIST_LABELS[dist],
            color=DIST_COLORS[dist],
            edgecolor="none",
        )

    labels = [f"{int(bins[i])}–{int(bins[i + 1])}" for i in range(NUM_HIST_BINS)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    algo_name = "ДП" if "dp" in source_key_prefix else "Жадный"
    mode_word = "весу" if mode == "weight" else "ценности"
    ax.set_title(
        f"Выбранные предметы ({algo_name}) по {mode_word}  (N = {items_counts[n_idx]:,})".replace(
            ",", " "
        ),
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Категория")
    ax.set_ylabel("Количество предметов")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.canvas.draw_idle()


def redraw_histogram_compare(ax, fig, active, n_idx, mode):
    """
    Гистограмма сравнения: жадный vs ДП для каждого распределения.
    Каждое распределение - пара столбцов (светлый = жадный, насыщенный = ДП).
    """
    ax.clear()
    wkey_gr = "greedy_picked_weights"
    pkey_gr = "greedy_picked_prices"
    wkey_dp = "dp_picked_weights"
    pkey_dp = "dp_picked_prices"

    gr_key = wkey_gr if mode == "weight" else pkey_gr
    dp_key = wkey_dp if mode == "weight" else pkey_dp

    if not active:
        _draw_empty(ax)
        fig.tight_layout()
        fig.canvas.draw_idle()
        return

    all_arrays = []
    for d in active:
        all_arrays.append(all_results[d][gr_key][n_idx])
        all_arrays.append(all_results[d][dp_key][n_idx])

    if all(a.size == 0 for a in all_arrays):
        ax.text(
            0.5,
            0.5,
            "Нет выбранных предметов",
            ha="center",
            va="center",
            fontsize=13,
            color="#6C7086",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw_idle()
        return

    bins = _compute_bins(all_arrays, mode)
    x = np.arange(NUM_HIST_BINS)

    # Каждое распределение даёт 2 столбца: жадный + ДП
    total_bars = len(active) * 2
    width = 0.7 / max(total_bars, 1)

    bar_idx = 0
    for dist in active:
        clr = DIST_COLORS[dist]
        lbl = DIST_LABELS[dist]

        gr_data = all_results[dist][gr_key][n_idx]
        dp_data = all_results[dist][dp_key][n_idx]

        gr_counts, _ = np.histogram(gr_data, bins=bins)
        dp_counts, _ = np.histogram(dp_data, bins=bins)

        offset_gr = (bar_idx - total_bars / 2 + 0.5) * width
        offset_dp = (bar_idx + 1 - total_bars / 2 + 0.5) * width

        ax.bar(
            x + offset_gr,
            gr_counts,
            width,
            label=f"{lbl} - Жадный",
            color=clr,
            alpha=0.5,
            edgecolor="none",
        )
        ax.bar(
            x + offset_dp,
            dp_counts,
            width,
            label=f"{lbl} - ДП",
            color=clr,
            alpha=1.0,
            edgecolor="none",
        )
        bar_idx += 2

    labels = [f"{int(bins[i])}–{int(bins[i + 1])}" for i in range(NUM_HIST_BINS)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    mode_word = "весу" if mode == "weight" else "ценности"
    ax.set_title(
        f"Сравнение выбранных предметов по {mode_word}  (N = {items_counts[n_idx]:,})".replace(
            ",", " "
        ),
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Категория")
    ax.set_ylabel("Количество предметов")
    ax.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.canvas.draw_idle()


def redraw_histogram_input(ax, fig, active, n_idx, mode):
    """Гистограмма входных данных (все сгенерированные предметы)."""
    ax.clear()
    if not active:
        _draw_empty(ax)
        fig.tight_layout()
        fig.canvas.draw_idle()
        return

    arrays = [
        item_arrays[d][n_idx]["weights" if mode == "weight" else "prices"]
        for d in active
    ]
    bins = _compute_bins(arrays, mode)
    x = np.arange(NUM_HIST_BINS)
    total = len(active)
    width = 0.7 / max(total, 1)

    for idx, dist in enumerate(active):
        data = arrays[idx]
        counts_arr, _ = np.histogram(data, bins=bins)
        offset = (idx - total / 2 + 0.5) * width
        ax.bar(
            x + offset,
            counts_arr,
            width,
            label=DIST_LABELS[dist],
            color=DIST_COLORS[dist],
            edgecolor="none",
        )

    labels = [f"{int(bins[i])}–{int(bins[i + 1])}" for i in range(NUM_HIST_BINS)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    mode_word = "весу" if mode == "weight" else "ценности"
    ax.set_title(
        f"Все предметы по {mode_word}  (N = {items_counts[n_idx]:,})".replace(",", " "),
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Категория")
    ax.set_ylabel("Количество предметов")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.canvas.draw_idle()


# UI
def main(page: ft.Page):
    page.title = "Задача о рюкзаке"
    page.window.resizable = True
    page.window.width = 1440
    page.window.height = 900
    page.padding = 0
    page.spacing = 0
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        color_scheme_seed="#89B4FA",
        font_family="Segoe UI",
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed="#89B4FA",
        font_family="Segoe UI",
    )

    is_dark = [True]

    # Состояние чекбоксов на вкладках
    tab_active_dists = {
        0: set(Distributions),
        1: set(Distributions),
        2: set(Distributions),
    }
    tab_checkboxes = {0: [], 1: [], 2: []}
    # Состояние гистограмм
    tab_hist_n_idx = {0: 0, 1: 0, 2: 0}
    tab_hist_mode = {0: "weight", 1: "weight", 2: "weight"}
    tab_hist_source = {0: "picked", 1: "picked", 2: "picked"}

    tab_bar_n_idx = {0: 0, 1: 0}  # индекс N для bar chart на вкладках ДП и Жадный

    def _on_bar_n_change(e):
        tab = e.control.data
        tab_bar_n_idx[tab] = int(e.control.value)
        active = sorted(tab_active_dists[tab], key=lambda d: d.value)
        if tab == 0:
            redraw_bar_single_n(
                ax_dp_items,
                fig_dp_items,
                active,
                "ds_counts",
                tab_bar_n_idx[0],
                "Взято предметов",
                "Число взятых предметов - Точный (ДП)",
            )
        elif tab == 1:
            redraw_bar_single_n(
                ax_gr_items,
                fig_gr_items,
                active,
                "greedy_counts",
                tab_bar_n_idx[1],
                "Взято предметов",
                "Число взятых предметов - Жадный",
            )
        page.update()

    def make_bar_n_dropdown(tab_index):
        dd = ft.Dropdown(
            label="Количество предметов (N)",
            value="0",
            width=220,
            data=tab_index,
            on_select=_on_bar_n_change,
            options=[
                ft.DropdownOption(key=str(i), text=f"N = {n:,}".replace(",", " "))
                for i, n in enumerate(items_counts)
            ],
            border_radius=10,
            dense=True,
        )
        return ft.Container(
            content=ft.Row([dd], spacing=16),
            padding=ft.Padding.symmetric(horizontal=16, vertical=10),
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

    fig_dp_time, ax_dp_time = plt.subplots(figsize=(9, 4))
    fig_dp_value, ax_dp_value = plt.subplots(figsize=(9, 4))
    fig_dp_items, ax_dp_items = plt.subplots(figsize=(9, 4))

    fig_gr_time, ax_gr_time = plt.subplots(figsize=(9, 4))
    fig_gr_value, ax_gr_value = plt.subplots(figsize=(9, 4))
    fig_gr_items, ax_gr_items = plt.subplots(figsize=(9, 4))
    fig_gr_error, ax_gr_error = plt.subplots(figsize=(9, 4))

    fig_cmp_time, axes_cmp_time = plt.subplots(1, 2, figsize=(14, 4.5))
    fig_cmp_values, axes_cmp_values = plt.subplots(1, 2, figsize=(14, 4.5))
    fig_cmp_errors, ax_cmp_errors = plt.subplots(figsize=(10, 4.5))
    fig_weight_cmp, axes_weight_cmp = plt.subplots(1, 2, figsize=(14, 4.5))

    # Фигуры гистограмм (по одной на вкладку)
    fig_hist_tab = {}
    ax_hist_tab = {}
    chart_hist_tab = {}
    for t in range(3):
        f, a = plt.subplots(figsize=(10, 4))
        fig_hist_tab[t] = f
        ax_hist_tab[t] = a
        chart_hist_tab[t] = fch.MatplotlibChart(figure=f, expand=True)

    # MatplotlibChart виджеты
    chart_dp_time = fch.MatplotlibChart(figure=fig_dp_time, expand=True)
    chart_dp_value = fch.MatplotlibChart(figure=fig_dp_value, expand=True)
    chart_dp_items = fch.MatplotlibChart(figure=fig_dp_items, expand=True)

    chart_gr_time = fch.MatplotlibChart(figure=fig_gr_time, expand=True)
    chart_gr_value = fch.MatplotlibChart(figure=fig_gr_value, expand=True)
    chart_gr_items = fch.MatplotlibChart(figure=fig_gr_items, expand=True)
    chart_gr_error = fch.MatplotlibChart(figure=fig_gr_error, expand=True)

    chart_cmp_time = fch.MatplotlibChart(figure=fig_cmp_time, expand=True)
    chart_cmp_values = fch.MatplotlibChart(figure=fig_cmp_values, expand=True)
    chart_cmp_errors = fch.MatplotlibChart(figure=fig_cmp_errors, expand=True)
    chart_weight_cmp = fch.MatplotlibChart(figure=fig_weight_cmp, expand=True)

    # Контейнеры для таблиц
    dp_tables_container = ft.Column([], scroll=ft.ScrollMode.AUTO)
    greedy_tables_container = ft.Column([], scroll=ft.ScrollMode.AUTO)

    # UI-утилиты
    def section_title(text, icon=None):
        controls = []
        if icon:
            controls.append(ft.Icon(icon, size=20, color=ft.Colors.PRIMARY))
        controls.append(
            ft.Text(text, size=17, weight=ft.FontWeight.W_600, color=ft.Colors.PRIMARY)
        )
        return ft.Container(
            content=ft.Row(controls, spacing=8),
            padding=ft.Padding.only(left=4, top=16, bottom=8),
        )

    def chart_card(chart_widget, height=380):
        return ft.Container(
            content=chart_widget,
            height=height,
            padding=12,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=8,
                color=ft.Colors.with_opacity(0.15, ft.Colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
        )

    def filter_panel(checkbox_row):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Фильтр распределений",
                        size=12,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                        weight=ft.FontWeight.W_500,
                    ),
                    checkbox_row,
                ],
                spacing=6,
            ),
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

    # Построение таблиц
    def make_table_card(dist, columns_def, row_builder):
        r = all_results[dist]
        rows = []
        for i, count in enumerate(r["items_counts"]):
            rows.append(row_builder(r, i, count))

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                DIST_ICONS[dist], color=DIST_FLET_COLORS[dist], size=18
                            ),
                            ft.Text(
                                DIST_LABELS[dist],
                                size=15,
                                weight=ft.FontWeight.W_600,
                                color=DIST_FLET_COLORS[dist],
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.DataTable(
                        columns=columns_def,
                        rows=rows,
                        heading_row_color=ft.Colors.with_opacity(
                            0.04, ft.Colors.ON_SURFACE
                        ),
                        data_row_max_height=42,
                        column_spacing=24,
                    ),
                ],
                spacing=8,
            ),
            padding=16,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

    # Dropdown-ы для гистограмм (по одному комплекту на вкладку)
    hist_n_dropdowns = {}
    hist_mode_dropdowns = {}

    def _on_hist_n_change(e):
        tab = e.control.data
        tab_hist_n_idx[tab] = int(e.control.value)
        active = sorted(tab_active_dists[tab], key=lambda d: d.value)
        _redraw_hist_for_tab(tab, active)
        page.update()

    def _on_hist_source_change(e):
        tab = e.control.data
        tab_hist_source[tab] = e.control.value
        active = sorted(tab_active_dists[tab], key=lambda d: d.value)
        _redraw_hist_for_tab(tab, active)
        page.update()

    def _on_hist_mode_change(e):
        tab = e.control.data
        tab_hist_mode[tab] = e.control.value
        active = sorted(tab_active_dists[tab], key=lambda d: d.value)
        _redraw_hist_for_tab(tab, active)
        page.update()

    def make_hist_controls(tab_index):
        dd_n = ft.Dropdown(
            label="Количество предметов (N)",
            value="0",
            width=220,
            data=tab_index,
            on_select=_on_hist_n_change,
            options=[
                ft.DropdownOption(key=str(i), text=f"N = {n:,}".replace(",", " "))
                for i, n in enumerate(items_counts)
            ],
            border_radius=10,
            dense=True,
        )
        dd_mode = ft.Dropdown(
            label="Показать по",
            value="weight",
            width=200,
            data=tab_index,
            on_select=_on_hist_mode_change,
            options=[
                ft.DropdownOption(key="weight", text="По весу"),
                ft.DropdownOption(key="price", text="По ценности"),
            ],
            border_radius=10,
            dense=True,
        )
        dd_source = ft.Dropdown(
            label="Источник",
            value="picked",
            width=240,
            data=tab_index,
            on_select=_on_hist_source_change,
            options=[
                ft.DropdownOption(key="picked", text="Выбранные алгоритмом"),
                ft.DropdownOption(key="input", text="Все предметы"),
            ],
            border_radius=10,
            dense=True,
        )
        hist_n_dropdowns[tab_index] = dd_n
        hist_mode_dropdowns[tab_index] = dd_mode
        return ft.Container(
            content=ft.Row([dd_n, dd_mode, dd_source], spacing=16, wrap=True),
            padding=ft.Padding.symmetric(horizontal=16, vertical=10),
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

    def rebuild_dp_tables(active):
        dp_tables_container.controls.clear()
        cols = [
            ft.DataColumn(ft.Text("N", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Время", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Сумм. ценность", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Взято", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Заполненность", weight=ft.FontWeight.W_600)),
        ]
        for dist in active:

            def row_builder(r, i, count, _r=all_results[dist]):
                total_w = _r["sum_item_weights"][i]
                dp_w = _r["sum_dp_weights"][i]
                fill = (dp_w / cCapacity * 100) if cCapacity > 0 else 0
                return ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(count))),
                        ft.DataCell(ft.Text(f"{_r['time_ds'][i]:.6f} с")),
                        ft.DataCell(ft.Text(str(_r["ds_values"][i]))),
                        ft.DataCell(ft.Text(str(_r["ds_counts"][i]))),
                        ft.DataCell(ft.Text(f"{fill:.1f}%")),
                    ]
                )

            dp_tables_container.controls.append(
                make_table_card(dist, cols, row_builder)
            )

    def rebuild_greedy_tables(active):
        greedy_tables_container.controls.clear()
        cols = [
            ft.DataColumn(ft.Text("N", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Время", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Ценность (оптим.)", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Ценность (базов.)", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Погр. оптим.", weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Погр. базов.", weight=ft.FontWeight.W_600)),
        ]
        for dist in active:

            def row_builder(r, i, count, _r=all_results[dist], _dist=dist):
                return ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(count))),
                        ft.DataCell(ft.Text(f"{_r['time_greedy'][i]:.6f} с")),
                        ft.DataCell(ft.Text(str(_r["greedy_values"][i]))),
                        ft.DataCell(ft.Text(str(basic_greedy_values[_dist][i]))),
                        ft.DataCell(ft.Text(f"{_r['errors'][i]:.2f}%")),
                        ft.DataCell(ft.Text(f"{basic_greedy_errors[_dist][i]:.2f}%")),
                    ]
                )

            greedy_tables_container.controls.append(
                make_table_card(dist, cols, row_builder)
            )

    # Перерисовка вкладки

    def _redraw_hist_for_tab(tab, active):
        n_idx = tab_hist_n_idx[tab]
        mode = tab_hist_mode[tab]
        source = tab_hist_source[tab]
        a = ax_hist_tab[tab]
        f = fig_hist_tab[tab]

        if source == "input":
            redraw_histogram_input(a, f, active, n_idx, mode)
        elif tab == 0:
            redraw_histogram_single(a, f, active, n_idx, mode, "dp_picked")
        elif tab == 1:
            redraw_histogram_single(a, f, active, n_idx, mode, "greedy_picked")
        elif tab == 2:
            redraw_histogram_compare(a, f, active, n_idx, mode)

    def rebuild_tab(tab, active):
        if tab == 0:
            rebuild_dp_tables(active)
            redraw_line(
                ax_dp_time,
                fig_dp_time,
                active,
                "time_ds",
                "Количество предметов (N)",
                "Время (сек)",
                "Время работы - Точный (ДП)",
                marker="s",
            )
            redraw_line(
                ax_dp_value,
                fig_dp_value,
                active,
                "ds_values",
                "N",
                "Сумм. ценность",
                "Максимальная ценность - Точный (ДП)",
                marker="s",
            )
            redraw_bar_single_n(
                ax_dp_items,
                fig_dp_items,
                active,
                "ds_counts",
                tab_bar_n_idx[0],
                "Взято предметов",
                "Число взятых предметов - Точный (ДП)",
            )
            _redraw_hist_for_tab(tab, active)
        elif tab == 1:
            rebuild_greedy_tables(active)
            redraw_line(
                ax_gr_time,
                fig_gr_time,
                active,
                "time_greedy",
                "Количество предметов (N)",
                "Время (сек)",
                "Время работы - Жадный",
            )
            redraw_line(
                ax_gr_value,
                fig_gr_value,
                active,
                "greedy_values",
                "N",
                "Сумм. ценность",
                "Максимальная ценность - Жадный",
            )
            redraw_bar_single_n(
                ax_gr_items,
                fig_gr_items,
                active,
                "greedy_counts",
                tab_bar_n_idx[1],
                "Взято предметов",
                "Число взятых предметов - Жадный",
            )
            redraw_line(
                ax_gr_error,
                fig_gr_error,
                active,
                "errors",
                "N",
                "Погрешность (%)",
                "Относительная погрешность жадного",
            )
            _redraw_hist_for_tab(tab, active)
        elif tab == 2:
            redraw_compare_pair(
                axes_cmp_time,
                fig_cmp_time,
                active,
                "time_greedy",
                "time_ds",
                "Время (сек)",
                "Время - Жадный",
                "Время - ДП",
            )
            redraw_compare_pair(
                axes_cmp_values,
                fig_cmp_values,
                active,
                "greedy_values",
                "ds_values",
                "Ценность",
                "Сумм. ценность - Жадный",
                "Сумм. ценность - ДП",
            )
            redraw_line(
                ax_cmp_errors,
                fig_cmp_errors,
                active,
                "errors",
                "N",
                "Погрешность (%)",
                "Относительная погрешность жадного - сравнение",
            )
            redraw_compare_pair(
                axes_weight_cmp,
                fig_weight_cmp,
                active,
                "sum_item_weights",
                "sum_dp_weights",
                "Вес",
                "Суммарный вес всех предметов",
                "Вес набора (ДП)",
            )
            _redraw_hist_for_tab(tab, active)

    # Чекбоксы

    def on_checkbox_change(e):
        tab = e.control.data["tab"]
        dist = e.control.data["dist"]
        if e.control.value:
            tab_active_dists[tab].add(dist)
        else:
            tab_active_dists[tab].discard(dist)
        active = sorted(tab_active_dists[tab], key=lambda d: d.value)
        rebuild_tab(tab, active)
        page.update()

    def make_checkbox_row(tab_index):
        cbs = []
        for dist in Distributions:
            cbs.append(
                ft.Checkbox(
                    label=DIST_LABELS[dist],
                    value=True,
                    data={"tab": tab_index, "dist": dist},
                    on_change=on_checkbox_change,
                    active_color=DIST_FLET_COLORS[dist],
                    check_color=ft.Colors.WHITE,
                )
            )
        tab_checkboxes[tab_index] = cbs
        return ft.Row(cbs, spacing=16, wrap=True)

    # Страницы контента

    page_exact = ft.Column(
        [
            filter_panel(make_checkbox_row(0)),
            section_title("Таблицы результатов", ft.Icons.TABLE_CHART),
            dp_tables_container,
            section_title("Графики", ft.Icons.SHOW_CHART),
            chart_card(chart_dp_time),
            ft.Container(height=12),
            chart_card(chart_dp_value),
            ft.Container(height=12),
            make_bar_n_dropdown(0),
            chart_card(chart_dp_items),
            ft.Container(height=12),
            section_title("Гистограмма входных данных", ft.Icons.BAR_CHART),
            make_hist_controls(0),
            ft.Container(height=8),
            chart_card(chart_hist_tab[0]),
            ft.Container(height=12),
        ],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=8,
        visible=True,
    )

    page_greedy = ft.Column(
        [
            filter_panel(make_checkbox_row(1)),
            section_title("Таблицы результатов", ft.Icons.TABLE_CHART),
            greedy_tables_container,
            section_title("Графики", ft.Icons.SHOW_CHART),
            chart_card(chart_gr_time),
            ft.Container(height=12),
            chart_card(chart_gr_value),
            ft.Container(height=12),
            make_bar_n_dropdown(1),
            chart_card(chart_gr_items),
            ft.Container(height=12),
            chart_card(chart_gr_error),
            section_title("Гистограмма входных данных", ft.Icons.BAR_CHART),
            make_hist_controls(1),
            ft.Container(height=8),
            chart_card(chart_hist_tab[1]),
            ft.Container(height=12),
        ],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=8,
        visible=False,
    )

    page_compare = ft.Column(
        [
            filter_panel(make_checkbox_row(2)),
            section_title("Время работы", ft.Icons.TIMER),
            chart_card(chart_cmp_time, height=420),
            ft.Container(height=12),
            section_title("Максимальная ценность", ft.Icons.DIAMOND),
            chart_card(chart_cmp_values, height=420),
            ft.Container(height=12),
            section_title("Относительная погрешность", ft.Icons.ERROR_OUTLINE),
            chart_card(chart_cmp_errors, height=420),
            section_title("Гистограмма входных данных", ft.Icons.BAR_CHART),
            make_hist_controls(2),
            ft.Container(height=8),
            chart_card(chart_hist_tab[2]),
            ft.Container(height=12),
        ],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=8,
        visible=False,
    )

    # Первичная отрисовка
    for tab_idx in range(3):
        rebuild_tab(tab_idx, sorted(tab_active_dists[tab_idx], key=lambda d: d.value))

    # Навигация (NavigationRail)
    pages_map = {0: page_exact, 1: page_greedy, 2: page_compare}
    page_titles = {
        0: "Точный алгоритм (ДП)",
        1: "Жадный алгоритм",
        2: "Сводное сравнение",
    }

    title_text = ft.Text(page_titles[0], size=22, weight=ft.FontWeight.BOLD)

    def on_nav_change(e):
        idx = e.control.selected_index
        for p in pages_map.values():
            p.visible = False
        pages_map[idx].visible = True
        title_text.value = page_titles[idx]
        page.update()

    # Переключение темный/светлый режим

    theme_icon = ft.IconButton(
        icon=ft.Icons.LIGHT_MODE,
        tooltip="Переключить тему",
        icon_color=ft.Colors.ON_SURFACE_VARIANT,
    )

    def toggle_theme(e):
        if is_dark[0]:
            page.theme_mode = ft.ThemeMode.LIGHT
            theme_icon.icon = ft.Icons.DARK_MODE
            apply_mpl_style(dark=False)
        else:
            page.theme_mode = ft.ThemeMode.DARK
            theme_icon.icon = ft.Icons.LIGHT_MODE
            apply_mpl_style(dark=True)
        is_dark[0] = not is_dark[0]
        for tab_idx in range(3):
            rebuild_tab(
                tab_idx, sorted(tab_active_dists[tab_idx], key=lambda d: d.value)
            )
        page.update()

    theme_icon.on_click = toggle_theme

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        min_extended_width=200,
        group_alignment=-0.9,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        indicator_color=ft.Colors.PRIMARY_CONTAINER,
        on_change=on_nav_change,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.CHECK_CIRCLE_OUTLINE,
                selected_icon=ft.Icons.CHECK_CIRCLE,
                label="Точный",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.FLASH_ON_OUTLINED,
                selected_icon=ft.Icons.FLASH_ON,
                label="Жадный",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.COMPARE_ARROWS_OUTLINED,
                selected_icon=ft.Icons.COMPARE_ARROWS,
                label="Сравнение",
            ),
        ],
    )

    # Заголовок

    header = ft.Container(
        content=ft.Row(
            [
                ft.Column(
                    [title_text],
                    spacing=2,
                    expand=True,
                ),
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.Icon(
                                            ft.Icons.INVENTORY_2,
                                            size=16,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                        ft.Text(
                                            f"W = {cCapacity}",
                                            size=12,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                    ],
                                    spacing=6,
                                ),
                                padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                                border_radius=20,
                                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            ),
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.Icon(
                                            ft.Icons.NUMBERS,
                                            size=16,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                        ft.Text(
                                            f"N: 1 … {items_counts[-1]:,}".replace(
                                                ",", " "
                                            ),
                                            size=12,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                    ],
                                    spacing=6,
                                ),
                                padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                                border_radius=20,
                                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            ),
                            theme_icon,
                        ],
                        spacing=8,
                    ),
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.symmetric(horizontal=24, vertical=16),
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        border=ft.Border.only(bottom=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)),
    )

    # Основной layout

    content_area = ft.Container(
        content=ft.Column(
            [
                header,
                ft.Container(
                    content=ft.Stack(
                        [page_exact, page_greedy, page_compare],
                        expand=True,
                    ),
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                ),
            ],
            spacing=0,
            expand=True,
        ),
        expand=True,
    )

    main_layout = ft.Row(
        [
            rail,
            ft.VerticalDivider(width=1, color=ft.Colors.OUTLINE_VARIANT),
            content_area,
        ],
        spacing=0,
        expand=True,
    )

    page.add(main_layout)


ft.run(main)
