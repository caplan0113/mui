import matplotlib.pyplot as plt
import numpy as np
from lib import *
from matplotlib.backends.backend_pdf import PdfPages
import math
from scipy import stats
import itertools
import seaborn as sns

TITLE = {"obj": "Eye tracking", "pos": "Camera Position", "rot": "Camera Rotation", "bpm": "Heart rate", "trigger": "Button", "state": "State", "subTask": "Subtask", "warning": "Warning", "collision": "Collision"}
UI = ["Audiovisual", "Visual", "Auditory", "None"]
OBJ_LABEL = ["Robot", "Arrow", "TaskPanel", "Shelf", "Building", "None"]
OBJ_COLOR = {"Robot": "red", "Arrow": "blue", "TaskPanel": "yellow", "Shelf": "green", "Building": "grey", "None": "grey"}
PATH = "pic/{0}/{1:02d}_{2:01d}.png"
FIG = "pic/graph_{0}.svg"
PNG = "pic/{0}.png"
DPI = 300
SAVE = False
FIGSIZE = (11.69, 8.27)
PPATH = "pdf/{0}.pdf"
TASK_TILE = [ # X, Z
    (23, 24),
    (20, 35),
    (20, 12),
    (13.5, 24),
    (10, 35),
    (10, 12),
    (7, 24),
    (20, 35),
    (20, 12),
    (10, 35),
    (10, 12)
]
TASK_TILE_CENTER = [
    (23.5, 0, 24.5),
    (20.5, 0, 35.5),
    (20.5, 0, 12.5),
    (14, 0, 24.5),
    (10.5, 0, 35.5),
    (10.5, 0, 12.5),
    (7.5, 0, 24.5),
    (20.5, 0, 35.5),
    (20.5, 0, 12.5),
    (10.5, 0, 35.5),
    (10.5, 0, 12.5)
]

TASK_LIM = [ # rot, (xmin, xmax), (zmin, zmax)
    (0, (20 ,24), (21, 28)),
    (1, (17, 24), (32, 36)),
    (2, (17, 24), (12, 16)),
    (3, (13.5, 17.5), (21, 28)),
    (1, (7, 14), (32, 36)),
    (2, (7, 14), (12, 16)),
    (3, (7, 11), (21, 28)),
    (1, (17, 24), (32, 36)),
    (2, (17, 24), (12, 16)),
    (1, (7, 14), (32, 36)),
    (2, (7, 14), (12, 16))
]

TASK_ROBOT = [
    [],
    [[("red", (0, 1), (3/4, 1))], [("red", (0, 1), (0, 1/4))]],
    [[("red", (0, 1), (3/4, 1)), ("magenta", (3/7, 4/7), (0, 3.5/4)), ("magenta", (6/7, 1), (0, 3.5/4))], [("red", (0, 1.5/7), (3/4, 1)), ("magenta", (6/7, 1), (0, 3.5/4))]],
    [[("red", (0, 1), (3/4, 1)), ("magenta", (3/7, 4/7), (0, 3.5/4)), ("magenta", (6/7, 1), (0, 3.5/4)), ("orange", (0, 1), (0.5/4, 1.5/4))], [("red", (0, 1.5/7), (3/4, 1)), ("magenta", (6/7, 1), (0, 3.5/4)), ("orange", (0, 1), (0.5/4, 1.5/4))]]
]

# graph functions *****************************
ROBOT_LABEL = ["No robot", "One robot", "Two robots", "Three robots"]

def get_collision():
    data = [[[] for _ in range(4)] for _ in range(3)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[ROBOT_NUM[sub["state"]]-1][uiId].append(sub["taskCollision"]) # robot 1-3
    
    return data, ROBOT_LABEL[1:4], UI

def get_time():
    data = [[[] for _ in range(4)] for _ in range(4)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[ROBOT_NUM[sub["state"]]][uiId].append(sub["taskTime"]) # robot 1-3
                else:
                    data[0][uiId].append(sub["taskTime"])
    
    return data, ROBOT_LABEL, UI

def get_distance():
    data = [[[] for _ in range(4)] for _ in range(3)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                dist = min(sub["min_dist"])  # Limit distance to 6m
                if sub["collision_flag"]:
                    data[ROBOT_NUM[sub["state"]]-1][uiId].append(dist) # robot 1-3
    
    return data, ROBOT_LABEL[1:4], UI

def get_response_time():
    data = [[[] for _ in range(3)] for _ in range(3)]  # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(3):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if not sub["collision_flag"]: continue

                pw = 0
                pt = sub["time"][0]
                flag = False
                center = np.array(TASK_TILE_CENTER[sub["state"]])
                tmp = []
                for t, w, p, c in zip(sub["time"], sub["warning"], sub["pos"], sub["collision"]):
                    # if pw == 0 and w > 0:
                    if w > 0 and not flag:
                        pt = t
                        flag = True
                        # udata[ROBOT_NUM[sub["state"]]-1].append((pw, w, pt-sub["time"][0]))
                    
                    if flag and (distance(p, center) >= 0.707):
                        if t != pt:
                            tmp.append(t-pt) # robot 1-3
                        flag = False
                    
                    pw = w
                
                if tmp:
                    data[ROBOT_NUM[sub["state"]]-1][uiId].append(np.mean(tmp))  # robot 1-3
                else:
                    print(f"{userData[0][0]['userId']} {uiId} {sub['state']} no response time")
                    pt = sub["time"][0]
                    flag = False
                    center = np.array(TASK_TILE_CENTER[sub["state"]])
                    tmp = []
                    for t, w, p, c in zip(sub["time"], sub["warning"], sub["pos"], sub["collision"]):
                        if w > 0 and not flag:
                            pt = t
                            flag = True
                            # udata[ROBOT_NUM[sub["state"]]-1].append((pw, w, pt-sub["time"][0]))
                        
                        if flag and (distance(p, center) >= 0.707):
                            if t != pt:
                                tmp.append(t-pt) # robot 1-3
                            flag = False
                    
                    print(tmp)
    
    return data, ROBOT_LABEL[1:4], UI[0:3]

def collision_plot(figsize=FIGSIZE, save=SAVE):
    data, titles, ui = get_collision()
    
    if save:
        save = "collision"

    # box_3x1(data[0][1:4], title="The number of collision", ylabel="the number of collision", ylim=(-1, 20), yticks=range(0, 16, 3), save=save)
    box_plot(data, ylabel="The number of collision", ylim=(-1, 20), yticks=range(0, 16, 3), xticklabel=titles, legends=ui, save=save)

def time_plot(figsize=FIGSIZE, save=SAVE):
    data, titles, ui = get_time()

    if save:
        save = "time"

    # box_1x4(data[0], title="Task completion time", ylabel="task completion time [s]", ylim=(-10, 150), yticks=range(0, 130, 20), save=save)
    box_plot(data, ylabel="Task completion time [s]", ylim=(-10, 145), yticks=range(0, 130, 20), xticklabel=titles, legends=ui, save=save)
    
def robot_distance_plot(figsize=FIGSIZE, save=SAVE):
    data, titles, ui = get_distance()
    
    if save:
        save = "robot_distance"
    # box_3x1(data[0][1:4], title="The closest distance to robots", ylabel="the closest distance to robots [m]", ylim=(-0.4, 4), yticks=np.arange(0, 3.1, 0.5), save=save)
    box_plot(data, ylabel="The closest distance to robots [m]", ylim=(-0.4, 3.5), yticks=np.arange(0, 3.1, 0.5), xticklabel=titles, legends=ui, save=save)

def cognition_time_plot(figsize=FIGSIZE, save=SAVE):
    data, titles, ui = get_response_time()

    if save:
        save = "cognition_time"

    # box_3x1(data[0][1:4], title="Time to leave task area after warning", ylabel="time to leave task area after warning[s]", xticklabel=UI[0:3], ylim=(-0.5, 5), yticks=np.arange(0, 4.1, 1), save=save)
    box_plot(data, ylabel="Time to leave task area after warning [s]", ylim=(-0.5, 5), yticks=np.arange(0, 4.1, 1), xticklabel=titles, legends=ui, save=save)

def safe_plot(save=SAVE):
    data, titles, ui = zip(get_collision(), get_distance())
    ylims = [(-1, 20), (-0.4, 3.7)]
    yticks = [range(0, 16, 3), np.arange(0, 3.1, 0.5)]

    if save:
        save = "safe"

    box_plot_2x1(data, ylim=ylims, yticks=yticks, xticklabel=titles, legends=UI, y_labels=["The number of collision", "The closest distance to robots [m]"], save=save)

def efficiency_plot(save=SAVE):
    data, titles, ui = zip(get_time(), get_response_time())
    ylims = [(-10, 150), (-0.5, 5)]
    yticks = [range(0, 130, 20), np.arange(0, 4.1, 1)]

    if save:
        save = "efficiency"

    box_plot_2x1(data, ylim=ylims, yticks=yticks, xticklabel=titles, legends=UI, y_labels=["Task completion time [s]", "Time to leave task area after warning [s]"], save=save)

def questionnaire_plot(save=SAVE):
    attr = ["easy", "annoy", "safe"]
    plot_data = [list(subject[a][:, 0:3].T) if a != "safe" else list(subject[a].T) for a in attr]

    labels = ["Easy to understand", "Distraction", "Safety perception"]

    if save:
        save = "questionnaire"

    box_plot(plot_data, ylabel="Subject questionnaire", ylim=(0.5, 9), yticks=range(1, 8), xticklabel=labels, legends=UI, save=save)

def nasatlx_plot(save=SAVE):
    attr = [["mental", "physical", "temporal", "performance"], ["effort", "frustration", "score"]]
    plot_data = [[list(subject[a].T) for a in r]for r in attr]

    labels = [["Mental demand", "Physical demand", "Temporal demand", "Performance"], ["Effort", "Frustration", "Overall score"]]

    if save:
        save = "nasatlx"

    box_plot_2x1(plot_data, ylabel="NASA-TLX", ylim=[(-10, 150)]*2, yticks=[range(0, 101, 20)]*2, xticklabel=labels, legends=UI, save=save)

def box_plot(data, ylabel, ylim, yticks, xticklabel=False, legends=False, figsize=(12, 6), save=SAVE):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(hspace=0.2, left=0.085, right=0.99, top=0.9, wspace=0.04, bottom=0.06)

    boxes = []
    labels = []

    if not xticklabel:
        xticklabel = ["Robot 1", "Robot 2", "Robot 3"]
    
    if not legends:
        legends = UI
    
    colors = ["#FF7F0E", "#1F77B4", "#2CA02C", "#A9A9A9"]
    median_colors = ["#FF7F0E", "#1F77B4", "#2CA02C", "#6B6B6B"]
    
    spacing_factor = 1.5  # Adjust this factor to increase or decrease the spacing between groups
    mean_color = "#000096"  # Color for the mean marker
    median_color = "#DC00DC"  # Color for the median line
    xticks_positions = []
    first_pos = 0
    for i, d in enumerate(data):
        current_positions = [first_pos + j for j in range(len(d))]
        xticks_positions.append((current_positions[0]+current_positions[-1])/2)
        first_pos = current_positions[-1] + spacing_factor
        bp = ax.boxplot(d, 
                        positions=current_positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanprops={"markerfacecolor": mean_color, "markeredgecolor": mean_color},
                        medianprops={"color":median_color, "linewidth": 2})
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor("black")
        
        for i, median in enumerate(bp['medians']):
            median.set_color(median_colors[i])
            median.set_marker('o')
            median.set_markersize(5)

        for whisker in bp['whiskers']:
            whisker.set_color(color="black")
        for cap in bp['caps']:
            cap.set_color("black")
        for flier in bp['fliers']:
            flier.set(marker='o', color='black', alpha=0.7)
        
    
    for k, g in enumerate(legends):
        boxes.append(plt.Rectangle((0, 0), 1, 1, color=colors[k], ec="black", alpha=0.5))
        labels.append(g)
    
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticklabel, fontsize=18)
    # ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(boxes, labels, loc='lower center', ncol=len(legends), fontsize=18, bbox_to_anchor=(0.5, 1))

    fig.supylabel(ylabel, x=0.01, y=0.5, fontsize=18)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
        print(f"Saved box plot to {FIG.format(save)}")
    else:
        plt.show()
    plt.close()

def box_plot_2x1(data,  ylim, yticks, ylabel=False, y_labels=False, xticklabel=False, legends=False, figsize=(12, 10), save=SAVE):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    fig.subplots_adjust(hspace=0.13, left=0.08, right=0.99, top=0.94, wspace=0.04, bottom=0.06)

    boxes = []
    labels = []

    if not xticklabel:
        xticklabel = ["Robot 1", "Robot 2", "Robot 3"]
    
    if not legends:
        legends = UI
    
    colors = ["#FF7F0E", "#1F77B4", "#2CA02C", "#A9A9A9"]
    median_colors = ["#FF7F0E", "#1F77B4", "#2CA02C", "#6B6B6B"]
    
    spacing_factor = 1.5  # Adjust this factor to increase or decrease the spacing between groups
    mean_color = "#000096"  # Color for the mean marker
    median_color = "#DC00DC"  # Color for the median line
    
    for k, ax in enumerate(axs):
        xticks_positions = []
        first_pos = 0
        for i, d in enumerate(data[k]):
            current_positions = [first_pos + j for j in range(len(d))]
            xticks_positions.append((current_positions[0]+current_positions[-1])/2)
            first_pos = current_positions[-1] + spacing_factor
            bp = ax.boxplot(d, 
                            positions=current_positions,
                            widths=0.6,
                            patch_artist=True,
                            showmeans=True,
                            meanprops={"markerfacecolor": mean_color, "markeredgecolor": mean_color},
                            medianprops={"color":median_color, "linewidth": 2})
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_edgecolor("black")
            
            for i, median in enumerate(bp['medians']):
                median.set_color(median_colors[i])
                median.set_marker('o')
                median.set_markersize(5)

            for whisker in bp['whiskers']:
                whisker.set_color(color="black")
            for cap in bp['caps']:
                cap.set_color("black")
            for flier in bp['fliers']:
                flier.set(marker='o', color='black', alpha=0.7)
        
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticklabel[k], fontsize=18)
        if y_labels:
            ax.set_ylabel(y_labels[k], fontsize=16, labelpad=15)
            ax.yaxis.set_label_coords(-0.06, 0.5)
        ax.set_ylim(*ylim[k])
        ax.set_yticks(yticks[k])
        ax.tick_params(axis='y', labelsize=16)
        
    
    for k, g in enumerate(legends):
        boxes.append(plt.Rectangle((0, 0), 1, 1, color=colors[k], ec="black", alpha=0.5))
        labels.append(g)
    
    
    axs[0].legend(boxes, labels, loc='lower center', ncol=len(legends), fontsize=18, bbox_to_anchor=(0.5, 1))

    if ylabel:
        fig.supylabel(ylabel, x=0.01, y=0.5, fontsize=18)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
    else:
        plt.show()
    plt.close()

def all_heatmap_plot(save=SAVE):
    datas, yticks, xticks = zip(get_collision(), get_distance(), get_time(), get_response_time())
    titles = ["Average Collision Count", "Average Robot Distance", "Average Task Time", "Average Response Time"]
    filenames = ["collision", "robot_distance", "task_time", "response_time"]

    for data, title, xtick, ytick, filename in zip(datas, titles, xticks, yticks, filenames):
        plot_data = [[np.mean(np.array(d)) for d in r] for r in data]
        if save:
            save = filename+"_r-ui"
        heatmap_plot(plot_data, title=title, xticklabel=xtick, yticklabel=ytick, save=save)
    
    for data, title, xtick, ytick, filename in zip(datas, titles, xticks, yticks, filenames):
        plot_data = np.array([[np.mean(np.array(d)) for d in r] for r in data]).T
        if save:
            save = filename+"_ui-r"
        heatmap_plot(plot_data, title=title, xticklabel=ytick, yticklabel=xtick, save=save)

def heatmap_plot(data, title, xticklabel, yticklabel, figsize=(12, 12), save=SAVE):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(hspace=0.2, left=0.04, right=1, top=0.95, wspace=0.04, bottom=0.05)
    df = pd.DataFrame(data, index=yticklabel, columns=xticklabel)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Wistia", cbar=True, square=True, linewidths=0.5, linecolor='black', ax=ax, annot_kws={"color": "black", "fontsize": 30, "weight": "bold"})
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)

    # for text in ax.texts:
    #     text.set_fontsize(16)
    #     text.set_color('black')
    
    ax.set_title(title, fontsize=25)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        # plt.savefig(FIG.format("heatmap_"+save), dpi=DPI)
        plt.savefig(PNG.format("heatmap_"+save), dpi=DPI)
        print(f"Saved heatmap to {PNG.format('heatmap_'+save)}")
    else:
        plt.show()
    
    plt.close()

def samples_list():
    data = [get_collision(), get_distance(), get_time(), get_response_time()]
    filenames = ["collision", "robot_distance", "task_time", "response_time"]
    for d, f in zip(data, filenames):
        print(f"[{f}]")
        if f=="response_time":
            samples(d, True)
        else:
            samples(d, True)
        print()
    
    attr = ["easy", "annoy", "safe"]
    data = [list(subject[a][:, 0:3].T) if a != "safe" else list(subject[a].T) for a in attr]
    samples2(data, attr)

    attr = ["mental", "physical", "temporal", "performance", "effort", "frustration", "score"]
    data = [list(subject[r].T) for r in attr]
    samples2(data, attr)
    

def samples2(data, titles):
    UI = ["M", "V", "A", "N"]
    for k, d in enumerate(data):
        print(f"[{titles[k]}]", end="")
        result = samples_test_rel_list(d, n_parametric=True)
        c = "T"
        group = friedmanchisquare(*d)
        cg = "Friedman"
        
        print(f" | {cg}: X^2 = {group.statistic:.2f}, p = {group.pvalue:.1e}")

        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if result[i][j][0]:
                    print(f"{UI[i]}-{UI[j]}: ({c} = {result[i][j][3]:.1f}, p = {result[i][j][2]:.1e})")

        print()

def samples(data, flag):
    UI = ["M", "V", "A", "N"]
    titles = data[1]
    data = data[0]
    for k, d in enumerate(data):
        print(titles[k], end="")
        if flag:
            result = samples_test_rel_list(d, n_parametric=True)
            c = "T"
            group = friedmanchisquare(*d)
            cg = "Friedman"
        else:
            result = samples_test_ind_list(d, n_parametric=True)
            c = "U"
            group = kruskal(*d)
            cg = "Kruskal-Wallis"
        
        print(f" | {cg}: X^2 = {group.statistic:.2f}, p = {group.pvalue:.1e}")

        for s in range(len(d)):
            res = ND_test(d[s])
            print(f"{UI[s]}: statistic = {res[0]:.2f}, p-value = {res[1]:.1e}")
        print()

        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if result[i][j][0]:
                    print(f"{UI[i]}-{UI[j]}: ({c}={result[i][j][3]:.1f}, p={result[i][j][2]:.1e})")
        print()

if __name__ == "__main__":
    try:
        all = load_all()
        attr = load_subject_attr()
        subject = load_subject()

        # questionnaire_plot(save=True)
        # nasatlx_plot(save=True)
        # collision_plot(save=True)
        # robot_distance_plot(save=True)
        # time_plot(save=True)
        # cognition_time_plot(save=True)
        # all_heatmap_plot(save=True)

        samples_list()
        

        # efficiency_plot(save=False)
        # safe_plot(save=False)
        # questionnaire_plot(save=False)
        # robot_distance_plot(save=False)
        # time_plot(save=False)
        # cognition_time_plot(save=False)
        # nasatlx_plot(save=False)
        # all_heatmap_plot(save=False)

        pass
    except Exception as e:
        traceback.print_exc()