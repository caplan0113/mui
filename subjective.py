import matplotlib.pyplot as plt
import numpy as np
from lib import *
from matplotlib.backends.backend_pdf import PdfPages

from itertools import combinations

FIGSIZE = (8, 6)
PATH = "pic/subjective/{0}.png"
SAVE = False
UI_LABEL = ["VA", "VO", "AO", "NO"]

KEY = {"easy": "UI was easy to understand", "annoy": "UI was obstructive", "useful": "UI was helpful", "trust": "UI was trustworthy", 
         "notice": "Noticed robot approaching", "distance": "Perceived distance", "direction": "Perceived direction", 
         "safe": "Felt safe during subtask", "vr": "Experienced VR sickness", "avoid": "Avoiding robots", 
         "mental": "NASA-TLX: Mental", "physical": "NASA-TLX: Physical",
         "temporal": "NASA-TLX: Temporal", "performance": "NASA-TLX: Performance",
         "effort": "NASA-TLX: Effort", "frustration": "NASA-TLX: Frustration",
         "score": "NASA-TLX: Score"}

NO_NONE = {"easy", "annoy", "useful", "trust"}

RANK_KEY = ["easy", "annoy", "useful", "trust", "notice", "distance", "direction", "safe", "vr", "avoid"]

UI_SET = {"easy": "easy", "annoy": "annoy", "useful": "useful", "trust": "trust"}
NASA_TLX_SET = {"mental", "physical", "temporal", "performance", "effort", "frustration", "score"}

def get_lim(attr):
    if attr in NASA_TLX_SET:
        return (-10, 110)
    else:
        return (0.4, 7.6)

def get_tick(attr):
    if attr in NASA_TLX_SET:
        return np.arange(0, 101, 10)
    else:
        return np.arange(1, 8, 1)

def box_plot(attr, figsize=FIGSIZE, save=SAVE, pdf=None):
    plot_data = data[attr]
    if attr in NO_NONE:
        plot_data = plot_data[:, 0:3]
        tlabel = UI_LABEL[0:3]
    else:
        tlabel = UI_LABEL
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(plot_data, tick_labels=tlabel, patch_artist=True, meanline=True, showmeans=True, notch=False)
    ax.set_title(KEY[attr])
    ax.set_ylim(get_lim(attr))
    ax.set_yticks(get_tick(attr))

    res = samples_test_rel_list(plot_data.T, n_parametric=True)
    txt = ""
    for v in range(len(tlabel)):
        for u in range(v + 1, len(tlabel)):
            if res[v][u][0]:
                txt += f"{tlabel[v]}-{tlabel[u]}(p={res[v][u][2]:.3f}), "

    if txt:
        ax.set_xlabel("*: "+txt[:-2])
    
    if save:
        plt.savefig(PATH.format("box_"+attr), dpi=300)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def box_ranking_plot(attr, figsize=FIGSIZE, save=SAVE, pdf=None):
    plot_data = ranking[attr]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(plot_data, tick_labels=UI_LABEL, patch_artist=True, meanline=True, showmeans=True, notch=False)
    ax.set_title(KEY[attr])
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks(range(1, 5))
    ax.invert_yaxis()

    res = samples_test_rel_list(plot_data.T, n_parametric=True)
    txt = ""
    for v in range(4):
        for u in range(v + 1, 4):
            if res[v][u][0]:
                txt += f"{UI_LABEL[v]}-{UI_LABEL[u]}(p={res[v][u][2]:.3f}), "
    
    if txt:
        ax.set_xlabel("*: "+txt[:-2])
    
    if save:
        plt.savefig(PATH.format("box_ranking_"+attr), dpi=300)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def all_corr():
    corr_list = []
    for attr1, attr2 in combinations(list(KEY.keys()), 2):
        corr_list.append((attr1, attr2, get_corr(data[attr1], data[attr2], axis=1)))
    return corr_list

def scatter_plot(attr1, attr2, figsize=FIGSIZE, save=SAVE, pdf=None):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.subplots_adjust(hspace=0.25, bottom=0.05)
    corr = get_corr(data[attr1], data[attr2], axis=1)
    for i, ax in enumerate(axs.flat):
        x, y = data[attr1][:, i], data[attr2][:, i]
        ax.scatter(x, y, label=UI_LABEL[i])
        ax.set_title(f"{UI_LABEL[i]}: {corr[i]:.2f}")
        # ax.set_xlabel(KEY[attr1])
        # ax.set_ylabel(KEY[attr2])
        ax.set_xlim(get_lim(attr1))
        ax.set_ylim(get_lim(attr2))
        # ax.set_xticks([])
        # ax.set_yticks([])
        if not np.isnan(corr[i]):
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), color="red", linestyle="--")
    
    fig.suptitle(f"{attr1} vs {attr2}")
    if save:
        plt.savefig(PATH.format(attr1 + "_" + attr2), dpi=300)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def all_scatter_plot():
    pdf = PdfPages("pdf/subjective_scatter.pdf")
    for attr1, attr2 in combinations(list(KEY.keys()), 2):
        scatter_plot(attr1, attr2, pdf=pdf, save=False)
    pdf.close()
    print("All scatter plots saved to pdf/subjective_scatter.pdf")

def all_box_plot():
    pdf = PdfPages("pdf/subjective_box.pdf")
    for attr in KEY.keys():
        box_plot(attr, pdf=pdf, save=False)
    pdf.close()
    print("All box plots saved to pdf/subjective_box.pdf")

def all_box_ranking_plot():
    pdf = PdfPages("pdf/subjective_box_ranking.pdf")
    for attr in RANK_KEY:
        box_ranking_plot(attr, pdf=pdf, save=False)
    pdf.close()
    print("All box ranking plots saved to pdf/subjective_box_ranking.pdf")

if __name__ == "__main__":
    data = load_subject()
    ranking = load_subject_ranking()
    data = {k: v for k, v in data.items()}
    # print(get_corr(data["mental"], data["physical"], axis=1))
    all_box_plot()
    all_box_ranking_plot()

    # for r in all_corr():
    #     attr1, attr2, corr = r
    #     if np.any(np.abs(corr) > 0.8):
    #         print(f"{attr1} and {attr2}: {corr}")

    # print(len(all_corr()))

    # scatter_plot("mental", "physical")
    # all_scatter_plot()