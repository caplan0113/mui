import matplotlib.pyplot as plt
import numpy as np
from lib import *
from matplotlib.backends.backend_pdf import PdfPages
import math
from scipy import stats
import itertools

TITLE = {"obj": "Eye tracking", "pos": "Camera Position", "rot": "Camera Rotation", "bpm": "Heart rate", "trigger": "Button", "state": "State", "subTask": "Subtask", "warning": "Warning", "collision": "Collision"}
UI = ["Audiovisual", "Visual", "Audio", "None"]
OBJ_LABEL = ["Robot", "Arrow", "TaskPanel", "Shelf", "Building", "None"]
OBJ_COLOR = {"Robot": "red", "Arrow": "blue", "TaskPanel": "yellow", "Shelf": "green", "Building": "grey", "None": "grey"}
PATH = "pic/{0}/{1:02d}_{2:01d}.png"
FIG = "pic/{0}.svg"
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

# eye tracking plot *****************************

def pie_plot(id, uiId, attr, target = None):
    data = load_subtask(id, uiId)
    for i, sub in enumerate(data):
        labels, counts = np.unique(sub[attr], return_counts=True)
        dic = dict(zip(labels, counts))
        print(dic)
        if target:
            target.reverse()
            value = [dic.get(label, 0) for label in target]
        else:
            value = dic.values()
            target = dic.keys()
        
        plt.figure(figsize=(7, 6))
        plt.pie(value, labels=target, autopct='%1.1f%%', startangle=90)
        plt.title('{0}: ({1:02d}, {2:01d}, {3})'.format(attr, id, uiId, i))
        # plt.savefig(PIC.format("pie_subtask", id, uiId))
        print(dic)
        plt.show()
        plt.close()

def barh_plot(userId, uiId, attr, target):
    data = load_subtask(userId, uiId)
    tasks = [ "{0}-{1}*".format(i, ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0}-{1} ".format(i, ROBOT_NUM[d["state"]]) for i, d in enumerate(data)]
    data = [ dict( [(label, count) for label, count in zip(*np.unique(sub[attr], return_counts=True))]) for sub in data ]
    data = [ [dic.get(label, 0) for label in target] for dic in data]
    data = np.array([ l/sum(l)*100 for l in data])
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.xaxis.set_visible(False)  # hide the x-axis
    ax.set_xlim(0, data.sum(axis=1).max())  # scale x-axis
    ax.set_ylim(-1, len(tasks) -0.5)

    for i, (colname, color) in enumerate(zip(target, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(tasks, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c <= 1: continue
            ax.text(x, y, "{0:.1f}".format(c), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(target), bbox_to_anchor=(0, 0),
              loc='upper left', fontsize='small')
    
    plt.subplots_adjust(left=0.05, right=0.99)
    plt.title('{0}: (User-{1:02d}, UI-{2})'.format(TITLE[attr], userId, UI[uiId]))
    # plt.savefig(PIC.format(attr, "barh", userId, uiId))
    plt.show()
    plt.close()

def barh_plot_all(userId, ui, attr, target=OBJ_LABEL, figsize=FIGSIZE, save=SAVE, pdf=None):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    for uiId, ax in enumerate(axs):
        data = all[userId, uiId]
        tasks = [ "{0}-{1}*".format(i, ROBOT_NUM[d["state"]])  if d["collision_flag"] else "{0}-{1} ".format(i, ROBOT_NUM[d["state"]]) for i, d in enumerate(data)]
        data = [ dict( [(label, count) for label, count in zip(*np.unique(sub[attr], return_counts=True))]) for sub in data ]
        data = [ [dic.get(label, 0) for label in target] for dic in data]
        data = np.array([ l/sum(l)*100 for l in data])
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))
        
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.xaxis.set_visible(False)  # hide the x-axis
        ax.set_xlim(0, data.sum(axis=1).max())  # scale x-axis
        ax.set_ylim(-1, len(tasks) -0.5)

        for i, (colname, color) in enumerate(zip(target, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(tasks, widths, left=starts, height=0.5,
                    label=colname, color=color)
            xcenters = starts + widths / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if c <= 1: continue
                ax.text(x, y, "{0:.1f}".format(c), ha='center', va='center',
                        color=text_color)
        ax.text(-0.08, 0.5, UI[uiId], transform=ax.transAxes, fontsize=10, va='center')
    
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05, hspace=0.03)
    fig.suptitle('{0}: (User-{1:02d})'.format(TITLE[attr], userId))
    fig.legend(target, ncol=len(target), bbox_to_anchor=(0.5, 0), loc='lower center', fontsize='small')

    if save:
        plt.savefig(PATH.format(attr+"_barh", userId, 9), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

# time series plot *****************************

def robot_distance(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Robot Distance [m]", [data["min_dist"] for data in subtaskData], subtaskData, ylim=(0, 6), figsize=figsize, save=save, pdf=pdf)

def robot_distance_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save: str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), np.abs(data["min_dist"][n:] - data["min_dist"][:-n])]) for data in subtaskData ]
    time_series_plot(f"Robot Distance Diff [m] ({n} frame)", plot_data, subtaskData, ylim=(-0.05, 1), figsize=figsize, save=save, pdf=pdf)

def pupil_size(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Pupil Size", [data["lg_pupil_d"] for data in subtaskData], subtaskData, ylim=(0, 7), figsize=figsize, save=save, pdf=pdf)

def eye_open(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Eye Open", [data["lg_open"] for data in subtaskData], subtaskData, ylim=(-0.2, 1.2), figsize=figsize, save=save, pdf=pdf)

def pupil_pos_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save: str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), np.linalg.norm(data["lg_pupil_pos"][n:] - data["lg_pupil_pos"][:-n], axis=1)]) for data in subtaskData ]
    time_series_plot(f"Pupil Position Diff [m] ({n} frame)", plot_data, subtaskData, ylim=(-0.05, 0.5), figsize=figsize, save=save, pdf=pdf)

def rot_y(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Rotation_y", [data["rot"][:, 1] for data in subtaskData], subtaskData, figsize=figsize, save=save, pdf=pdf)

def rot_y_diff(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.abs(np.diff(data["rot"], axis=0, prepend=[data["rot"][0]]))[:, 1] for data in subtaskData ]
    time_series_plot("Rotation_y Diff [degree] (1 frame)", plot_data, subtaskData, ylim=(-2, 6), figsize=figsize, save=save, pdf=pdf)

def rot_y_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save: str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), np.abs(data["rot"][n:] - data["rot"][:-n])[:, 1]]) for data in subtaskData ]
    time_series_plot(f"Rotation_y Diff [degree] ({n} frame)", plot_data, subtaskData, ylim=(-1, 150), figsize=figsize, save=save, pdf=pdf)

def bpm(userId, uiId, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    time_series_plot("Heart Rate", [data["bpm"] for data in subtaskData], subtaskData, ylim=(60, 120), figsize=figsize, save=save, pdf=pdf)

def pos_x_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), (data["pos"][n:] - data["pos"][:-n])[:, 0]]) for data in subtaskData ]
    time_series_plot(f"Position X Diff [{n} frame]", plot_data, subtaskData, ylim=(-0.5, 0.5), figsize=figsize, save=save, pdf=pdf)

def pos_z_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), (data["pos"][n:] - data["pos"][:-n])[:, 2]]) for data in subtaskData ]
    time_series_plot(f"Position Z Diff [{n} frame]", plot_data, subtaskData, ylim=(-0.5, 0.5), figsize=figsize, save=save, pdf=pdf)

def pos_xz_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), distance(data["pos"][n:], data["pos"][:-n])]) for data in subtaskData ]
    time_series_plot(f"Position XZ Diff [m] ({n} frame)", plot_data, subtaskData, ylim=(-0.2, 2), figsize=figsize, save=save, pdf=pdf)

def pos_xz_diff_center(userId, uiId, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    plot_data = [distance(data["pos"], np.array(TASK_TILE_CENTER[data["state"]])) for data in subtaskData]
    time_series_plot("Position XZ Diff Center [m]", plot_data, subtaskData, ylim=(-0.5, 5), figsize=figsize, save=save, pdf=pdf)

def twin_xz_roty_diff_n(userId, uiId, n:int, name="", figsize=FIGSIZE, save:str =None, pdf=None, legends: list[str]=None):
    subtaskData = all[userId, uiId]
    plot_data = [ np.concatenate([np.full(n, 0), distance(data["pos"][n:], data["pos"][:-n])]) for data in subtaskData ]
    plot_data2 = [ np.concatenate([np.full(n, 0), np.abs(data["rot"][n:] - data["rot"][:-n])[:, 1]]) for data in subtaskData ]
    time_series_plot(f"Position XZ Diff [m] & Rotation Y Diff [degree] ({n} frame)"+name, plot_data, subtaskData, plot_data2=plot_data2, ylim=(-0.1, 0.3), ylim2=(-5, 15), legends=legends, figsize=figsize, save=save, pdf=pdf)

def twin_robot_distance_xz_diff_center(userId, uiId, name="", figsize=FIGSIZE, save:str =None, pdf=None, legends: list[str]=None):
    subtaskData = all[userId, uiId]
    plot_data = [ data["min_dist"] for data in subtaskData ]
    plot_data2 = [ distance(data["pos"], np.array(TASK_TILE_CENTER[data["state"]])) for data in subtaskData ]
    time_series_plot(f"Robot Distance & Position XZ Diff Center [m]"+name, plot_data, subtaskData, plot_data2=plot_data2, ylim=(0, 6), ylim2=(-0.5, 5), legends=legends, figsize=figsize, save=save, pdf=pdf)

def twin_rot_y_gaze(userId, uiId, name="", figsize=FIGSIZE, save:str =None, pdf=None, legends: list[str]=None):
    subtaskData = all[userId, uiId]
    plot_data = [ data["rot"][:, 1] for data in subtaskData ]
    plot_data2 = [data["lg_rot"][:, 1] for data in subtaskData]
    time_series_plot(f"Rotation Y & Gaze {name}", plot_data, subtaskData, plot_data2=plot_data2, legends=legends, figsize=figsize, save=save, pdf=pdf)

def gaze_diff_n(userId, uiId, n:int, figsize=FIGSIZE, save:str =None, pdf=None):
    subtaskData = all[userId, uiId]
    gaze_vec1 = [rot2vec(data["lg_rot"]) for data in subtaskData]
    plot_data1 = [ np.concatenate([np.full(n, 0), np.linalg.norm(vec[n:] - vec[:-n], axis=1)]) for vec in gaze_vec1]
    gaze_vec2 = [rot2vec(data["rg_rot"]) for data in subtaskData]
    plot_data2 = [ np.concatenate([np.full(n, 0), np.linalg.norm(vec[n:] - vec[:-n], axis=1)]) for vec in gaze_vec2]
    plot_data = []
    for pd1, pd2 in zip(plot_data1, plot_data2):
        pd = []
        for l, r in zip(pd1, pd2):
            if l >= 1.1 and r >= 1.1:
                pd.append(0)
            elif l >= 1.1:
                pd.append(r)
            elif r >= 1.1:
                pd.append(l)
            elif math.isclose(l, 0):
                pd.append(r)
            else:
                pd.append(l)
        
        plot_data.append(pd)

    time_series_plot(f"Gaze Diff [degree] ({n} frame)", plot_data, subtaskData, ylim=(-0.1, 1.0), figsize=figsize, save=save, pdf=pdf)

def warning_plot(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    window = 5
    subtaskData = all[userId, uiId]
    for i, sub in enumerate(subtaskData):
        masks = get_warning_mask(sub["warning_filter"], window=120)
        data_rot_y_diff_n = np.concatenate([np.full(window, 0), np.abs(sub["rot"][window:] - sub["rot"][:-window])[:, 1]])
        data_pos_xz_diff_n = np.concatenate([np.full(window, 0), distance(sub["pos"][window:], sub["pos"][:-window])])
        data_pos_xz_diff_center = distance(sub["pos"], np.array(TASK_TILE_CENTER[sub["state"]]))

        for p in range(4):
            for n in range(4):
                if len(masks[p][n]) == 0: continue

                fig, axs = plt.subplots(nrows=len(masks[p][n]), ncols=1, figsize=figsize)
                # fig.subplots_adjust(hspace=0.3, left=0, right=1, top=0.9, wspace=0, bottom=0.05)
                fig.suptitle("Warning: User-{0:02d}, UI-{1} ({2}), {3}: {4}-{5}".format(userId, UI[uiId], subtaskData[0]["taskOrder"], sub["label"], p, n))
                if len(masks[p][n]) == 1:
                    axs = [axs]
                else:
                    axs = axs.flatten()

                for ax, mask in zip(axs, masks[p][n]):
                    x = sub["time"][mask]
                    ax.set_xlim(x[0], x[-1])

                    # plot_rot_y = sub["rot"][mask][:, 1]
                    # ax.plot(x, plot_rot_y)

                    plot_rot_y_diff_n = data_rot_y_diff_n[mask]
                    ax.plot(x, plot_rot_y_diff_n)
                    ax.set_ylim(-5, 15)


                    warning_bgcolor(ax, x, sub["warning_filter"][mask])

                    ax2 = ax.twinx()
                    # ax2.set_ylim(-0.1, 0.3)
                    # ax2.set_ylabel("Position XZ Diff")
                    # plot_pos_xz_diff_n = data_pos_xz_diff_n[mask]
                    # ax2.plot(x, plot_pos_xz_diff_n, color="orange", alpha=0.7)

                    ax2.set_ylim(-0.5, 5)
                    plot_pos_xz_diff_ceter = data_pos_xz_diff_center[mask]
                    ax2.plot(x, np.full_like(x, plot_pos_xz_diff_ceter), color="green", alpha=0.7)
                
                plt.tight_layout()
                plt.show()
                plt.close()

# util *****************************

def time_series_fft(name:str, plot_data, subtaskData, plot_data2=None, legends: list[str]=None, ylim: tuple[float, float]=None, ylim2: tuple[float, float]=None, xlim:tuple[float, float]=None, figsize:tuple[float, float]=FIGSIZE, save: str=None, pdf=None):
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("{0}: User-{1:02d}, UI-{2} ({3})".format(name, subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]], subtaskData[0]["taskOrder"]))
    fig.subplots_adjust(hspace=0.55, left=0.09, right=0.955, top=0.93, wspace=0, bottom=0.07)
    
    lines = []
    for i, (freqs, power) in enumerate(plot_data):
        # axs[i].set_ylim(*ylim)
        # axs[i].set_xlim(*xlim)
        # xmax = max(x[i])
        # axs[i].set_xlim(0, xmax)
        axs[i].text(-0.09, 0.5, subtaskData[i]["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        line1,  = axs[i].plot(freqs[1:len(freqs)//2], power[1:len(freqs)//2])
        lines.append(line1)

    # plt.tight_layout()
    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, subtaskData[0]["userId"], subtaskData[1]["uiId"]), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def time_series_plot(name:str, plot_data, subtaskData, x=None, plot_data2=None, legends: list[str]=None, ylim: tuple[float, float]=None, ylim2: tuple[float, float]=None, xlim:tuple[float, float]=None, figsize:tuple[float, float]=FIGSIZE, save: str=None, pdf=None):
    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=figsize)
    fig.suptitle("{0}: User-{1:02d}, UI-{2} ({3})".format(name, subtaskData[0]["userId"], UI[subtaskData[0]["uiId"]], subtaskData[0]["taskOrder"]))
    fig.subplots_adjust(hspace=0.55, left=0.09, right=0.955, top=0.93, wspace=0, bottom=0.07)


    if not x:
        x = [ data["time"] - data["time"][0] for data in subtaskData ]

    if not xlim:
        xlim = (0, max([r[-1] for r in x]))
        xmax = xlim[1]
    else:
        xmax = xlim[1]
    # xmax = 40.0

    if not ylim:
        ymin, ymax = min([min(r) for r in plot_data]), max([max(r) for r in plot_data])
        dy = (ymax - ymin) / 10
        ylim = ymin-dy, ymax+dy
    
    lines = []
    for i, pd in enumerate(plot_data):
        axs[i].set_ylim(*ylim)
        # axs[i].set_xlim(*xlim)
        xmax = max(x[i])
        # xmax = 50
        axs[i].set_xlim(0, xmax)
        axs[i].text(-0.09, 0.5, subtaskData[i]["label"], transform=axs[i].transAxes, fontsize=10, va='center')
        line1,  = axs[i].plot(x[i], pd)
        lines.append(line1)
        obj_bgcolor(axs[i], x[i], subtaskData[i]["obj"])
        collision_plot(axs[i], x[i], subtaskData[i]["collision"], ylim)
        # collision_plot(axs[i], x[i], subtaskData[i]["trigger"], xmax)
        warning_bgcolor(axs[i], x[i], subtaskData[i]["warning_filter"])

        if plot_data2:
            ax2 = axs[i].twinx()

            if not ylim2:
                ymin2, ymax2 = min([min(r) for r in plot_data2]), max([max(r) for r in plot_data2])
                dy2 = (ymax2 - ymin2) / 10
                ylim2 = ymin2-dy2, ymax2+dy2
            ax2.set_ylim(*ylim2)
            line2, = ax2.plot(x[i], plot_data2[i], color="orange", alpha=0.7)
            lines.append(line2)

    if legends:
        fig.legend(handles=lines[0:2], labels=legends, ncol=len(legends), bbox_to_anchor=(0.5, 0), loc='lower center', fontsize='small')        

    # plt.tight_layout()
    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, subtaskData[0]["userId"], subtaskData[1]["uiId"]), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def obj_bgcolor(ax, x, obj_list):
    prev = obj_list[0]
    pt = x[0]
    for t, obj in zip(x, obj_list):
        if obj != prev:
            if prev == "Robot" or prev == "Arrow":
                ax.axvspan(pt, t, facecolor=OBJ_COLOR[prev], alpha=0.3)
            prev = obj
            pt = t
    if prev == "Robot" or prev == "Arrow":
        ax.axvspan(pt, x[-1], facecolor=OBJ_COLOR[prev], alpha=0.3)

def warning_bgcolor(ax, x, warning_list):
    prev = warning_list[0]
    pt = x[0]
    for t, obj in zip(x, warning_list):
        if obj != prev:
            if prev != 0:
                ax.axvspan(pt, t, 0, prev*0.15, facecolor="green", alpha=0.5)
            prev = obj
            pt = t
    if prev != 0:
        ax.axvspan(pt, x[-1], 0, prev*0.15, facecolor="green", alpha=0.5)

def collision_plot(ax, x, collision_list, ylim):
    y = ylim[1] - (ylim[1] - ylim[0]) * 0.05
    for t, collision in zip(x, collision_list):
        if collision:
            ax.scatter(t, y, marker="*", color="magenta", s=20, zorder=5)
            # ax.text(t/xmax, 0.99, "*", fontsize=15, color="magenta", ha='center', va='top', transform=ax.transAxes)

def calculate_change(userId, uiId, attr):
    data = load_subtask(userId, uiId)
    for i, sub in enumerate(data):
        changes = []
        sub = sub[attr]
        prev = np.array([0]*len(sub[0]))
        NONE_VEC = np.zeros(sub[0].shape)

        for vec in sub:
            if (vec == NONE_VEC).all(): continue

            if (prev != NONE_VEC).all():
                change = np.linalg.norm(vec - prev)
                changes.append(change)
            prev = vec

        total_change = sum(changes)
        average_change = total_change / len(changes) if changes else 0

        print("({0}, {1}, {2}) Total change: {3}, Average change: {4})".format(userId, uiId, i, total_change, average_change))

# all data save and show plot *****************************

def save_pdf(func, args=dict(), userIdRange=range(0, N), uiIdRange=range(0, 4), name=""):
    os.makedirs("pdf", exist_ok=True)
    pdf_path = PPATH.format(func.__name__ + name)
    pdf = PdfPages(pdf_path)
    try: 
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId, pdf=pdf, **args)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
        traceback.print_exc()
    pdf.close()
    print("Saved PDF: {0}".format(pdf_path))

def show_plot(func, args=dict(), userIdRange=range(0, N), uiIdRange=range(0, 4)):
    try:
        for userId in userIdRange:
            for uiId in uiIdRange:
                func(userId, uiId, **args)
    except Exception as e:
        print("Error in {0}: {1}".format(func.__name__, e))
        traceback.print_exc()

def save_pic(func, args=dict(), userIdRange=range(0, N), uiIdRange=range(0, 4)):
    for userId in userIdRange:
        for uiId in uiIdRange:
            try:
                func(userId, uiId, save=func.__name__, **args)
            except Exception as e:
                print("Error in {0}: {1}".format(func.__name__, e))
                traceback.print_exc()

def map_func(func, args=dict(), userIdRange=range(0, N), uiIdRange=range(0, 4)):
    for userId in userIdRange:
        for uiId in uiIdRange:
            try:
                func(userId, uiId, **args)
            except Exception as e:
                print("Error in {0}: {1}".format(func.__name__, e))
                traceback.print_exc()

# 2D map *****************************

def pos_xz(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    axs = axs.flatten()
    fig.subplots_adjust(hspace=0.3, left=0, right=1, top=0.9, wspace=0, bottom=0.05)
    fig.suptitle("Position XZ: User-{0:02d}, UI-{1} ({2})".format(userId, UI[uiId], subtaskData[0]["taskOrder"]))

    for i, ax in enumerate(axs):
        x, z = subtaskData[i]["pos"][:, 0], subtaskData[i]["pos"][:, 2]
        cx, cz = subtaskData[i]["pos"][:, 0][subtaskData[i]["collision"]], subtaskData[i]["pos"][:, 2][subtaskData[i]["collision"]]
        s, xlim, zlim = TASK_LIM[subtaskData[i]["state"]]
        if s == 0:
            px, py = z, x
            pcx, pcy = cz, cx
            x_lim = zlim; y_lim = xlim
            xlabel = "Z"; ylabel = "X"
        elif s == 1:
            px, py = x, z
            pcx, pcy = cx, cz
            x_lim = xlim; y_lim = zlim
            xlabel = "X"; ylabel = "Z"
        elif s == 2:
            px, py = x, z
            pcx, pcy = cx, cz
            x_lim = xlim; y_lim = zlim
            xlabel = "X"; ylabel = "Z"
        else:
            px, py = z, x
            pcx, pcy = cz, cx
            x_lim = zlim; y_lim = xlim
            xlabel = "Z"; ylabel = "X"
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xticks(np.arange(x_lim[0], x_lim[1]+1, 1))
        ax.set_yticks(np.arange(y_lim[0], y_lim[1]+1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        
        if s == 0:
            ax.invert_xaxis()
        elif s == 2:
            ax.invert_xaxis()
            ax.invert_yaxis()
        elif s == 3:
            ax.invert_yaxis()
        
        task_bgcolor(ax, x_lim, s, ROBOT_NUM[subtaskData[i]["state"]], subtaskData[i]["collision_flag"])

        mask0, mask1, mask2, mask3 = subtaskData[i]["warning"]==0, subtaskData[i]["warning"]==1, subtaskData[i]["warning"]==2, subtaskData[i]["warning"]==3
        px0, py0 = px[mask0], py[mask0]
        px1, py1 = px[mask1], py[mask1]
        px2, py2 = px[mask2], py[mask2]
        px3, py3 = px[mask3], py[mask3]
        ax.scatter(px0, py0, color="green", s=1, zorder=1)
        ax.scatter(px1, py1, color="gold", s=1, zorder=1)
        ax.scatter(px2, py2, color="darkorange", s=1, zorder=1)
        ax.scatter(px3, py3, color="red", s=1, zorder=1)
        
        # ax.scatter(px, py, zorder=1, s=1)
        ax.scatter(pcx, pcy, color="red", marker="*", s=50, zorder=2) # collision
        
        ax.set_title("Subtask {0}".format(subtaskData[i]["label"]))

    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, subtaskData[0]["userId"], subtaskData[1]["uiId"]), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def task_bgcolor(ax, xlim, s, robot_num, collision_flag):
    xmin, xmax = xlim
    ax.axvspan(xmin+3, xmax-3, 3/4, 1, facecolor="green", alpha=0.5)
    robots = TASK_ROBOT[robot_num][not collision_flag]

    for c, x, y in robots:
        x1, x2 = x; y1, y2 = y
        if s % 2 == 0:
            xs = xmax - (xmax - xmin) * x1
            xe = xmax - (xmax - xmin) * x2
        else:
            xs = xmin + (xmax - xmin) * x1
            xe = xmin + (xmax - xmin) * x2
        ax.axvspan(xs, xe, y1, y2, edgecolor=c, fill=False, hatch="/", alpha=0.5, lw=1)

# robot number plot *****************************
def subtask_time(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    userData = all[userId]
    
    collision = [[(ROBOT_NUM[data["state"]], data["taskTime"]) for data in subtaskData if data["collision_flag"]] for subtaskData in userData]
    no_collision = [[(ROBOT_NUM[data["state"]], data["taskTime"]) for data in subtaskData if not data["collision_flag"]] for subtaskData in userData]

    robot_num_plot("Subtask Time", collision, no_collision, userId, figsize=figsize, save=save, pdf=pdf)

def collision_count(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    userData = all[userId]
    
    collision = [[(ROBOT_NUM[data["state"]], data["taskCollision"]) for data in subtaskData if data["collision_flag"]] for subtaskData in userData]
    no_collision = [[(ROBOT_NUM[data["state"]], data["taskCollision"]) for data in subtaskData if not data["collision_flag"]] for subtaskData in userData]

    robot_num_plot("Collision Count", collision, no_collision, userId, figsize=figsize, save=save, pdf=pdf)

def mistake_count(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    collision = [[(ROBOT_NUM[data["state"]], data["taskMistake"]) for data in subtaskData if data["collision_flag"]] for subtaskData in all[userId]]
    no_collision = [[(ROBOT_NUM[data["state"]], data["taskMistake"]) for data in subtaskData if not data["collision_flag"]] for subtaskData in all[userId]]

    robot_num_plot("Count of Mistake", collision, no_collision, userId, figsize=figsize, save=save, pdf=pdf)

def robot_num_plot(name: str, collision, no_collision, userId, figsize=FIGSIZE, save=SAVE, pdf=None):
    userData = all[userId]

    ymax = max([max([y for _, y in sub]) for sub in collision + no_collision])
    dy = max(0.01, ymax/10)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.subplots_adjust(hspace=0.28, left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    fig.suptitle(name+": User-{0:02d}".format(userId))
    axs = axs.flatten()
    plot_col1 = 0; plcot_col2 = 0; plot_ncol = 0
    for i, ax in enumerate(axs):
        ax.set_title("UI-{0} ({1})".format(UI[i], userData[i][0]["taskOrder"]))
        ax.set_ylim(-dy, ymax+dy)
        ax.set_xlim(0.5, 3.5)
        ax.set_xticks(np.arange(1, 4, 1))
        # ax.set_yticks(np.arange(0, 21, 1))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.set_xlabel("Robot Num")
        # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        flag = [True]*4
        for x, y in collision[i]:
            if flag[x]:
                flag[x] = False
                plot_col1 = ax.scatter(x, y, marker="<", color="red", s=50)
            else:
                plot_col2 = ax.scatter(x, y, marker=">", color="red", s=50)

        for x, y in no_collision[i]:
            plot_ncol = ax.scatter(x, y, marker="o", color="blue", s=50, alpha=0.3)
    
    fig.legend([plot_col1, plot_col2, plot_ncol], ["Collision-pattern 1", "Collision-pattern 2", "No Collision-pattern"], ncol=3, bbox_to_anchor=(0.5, 0), loc='lower center', fontsize='small')

    # plt.tight_layout()
    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, userId, 9), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

# scatter plot *****************************

def scatter_pos_xz_diff_center_robot_distance(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    data1 = [distance(data["pos"], np.array(TASK_TILE_CENTER[data["state"]])) for data in subtaskData]
    data2 = [data["min_dist"] for data in subtaskData]

    scatter_plot(data1, data2, "Position XZ Diff Center & Robot Distance: User-{0:02d}, UI-{1} ({2})".format(userId, UI[uiId], subtaskData[0]["taskOrder"]), "Position XZ Diff Center [m]", "Robot Distance [m]", subtaskData, xlim=(0, 6), ylim=(0, 6), figsize=figsize, save=save, pdf=pdf)


def scatter_pupil_size_pos_xz_diff_center(userId, uiId, n:int, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    data1 = [data["lg_pupil_d"] for data in subtaskData]
    data2 = [distance(data["pos"], np.array(TASK_TILE_CENTER[data["state"]])) for data in subtaskData]

    scatter_plot(data1, data2, "Pupil Size Diff & Position XZ Diff Center [{0} frame]: User-{1:02d}, UI-{2} ({3})".format(n, userId, UI[uiId], subtaskData[0]["taskOrder"]), "Pupil Size Diff [mm]", "Position XZ Diff Center [m]", subtaskData, xlim=(-0.5, 7), ylim=(0, 6), figsize=figsize, save=save, pdf=pdf)

def scatter_pupil_d_robot_distance(userId, uiId, n: int, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    data1 = [slide(data["lg_pupil_d"], n) for data in subtaskData]
    data2 = [data["min_dist"] for data in subtaskData]

    scatter_plot(data1, data2, "Pupil Size Diff [slide:{0} frame] & Robot Distance: User-{1:02d}, UI-{2} ({3})".format(n, userId, UI[uiId], subtaskData[0]["taskOrder"]), "Pupil Size Diff [mm]", "Robot Distance [m]", subtaskData, xlim=(-0.5, 7), ylim=(0, 6), figsize=figsize, save=save, pdf=pdf)

def scatter_warning_pupil_d(userId, uiId, n: int, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    data1 = [slide(data["lg_pupil_d"], n) for data in subtaskData]
    data2 = [data["warning"] for data in subtaskData]

    scatter_plot(data1, data2, "Pupil Size Diff & Warning: User-{0:02d}, UI-{1} ({2})".format(userId, UI[uiId], subtaskData[0]["taskOrder"]), "Pupil Size Diff [mm]", "Warning", subtaskData, xlim=(-0.5, 7), ylim=(-0.3, 3.3), figsize=figsize, save=save, pdf=pdf)

def scatter_plot(data1, data2, title, xlabel, ylabel, subtaskData, xlim, ylim, figsize=FIGSIZE, save=SAVE, pdf=None):
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=figsize)

    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.9, wspace=0.11, bottom=0.08)

    axs = axs.flatten()

    if not xlim:
        xlim = (min([min(d) for d in data1]), max([max(d) for d in data1]))

    if not ylim:
        ylim = (min([min(d) for d in data2]), max([max(d) for d in data2]))
    
    corr = get_corr(data1, data2)

    for i, (ax, d1, d2, co) in enumerate(zip(axs, data1, data2, corr)):
        ax.set_title("Subtask {0}: {1:.2f}".format(subtaskData[i]["label"], co))
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        
        mask0, mask1, mask2, mask3 = subtaskData[i]["warning"]==0, subtaskData[i]["warning"]==1, subtaskData[i]["warning"]==2, subtaskData[i]["warning"]==3
        d1_0, d2_0 = d1[mask0], d2[mask0]
        d1_1, d2_1 = d1[mask1], d2[mask1]
        d1_2, d2_2 = d1[mask2], d2[mask2]
        d1_3, d2_3 = d1[mask3], d2[mask3]
        ax.scatter(d1_0, d2_0, color="green", s=5, label="Warning 0", alpha=0.7)
        ax.scatter(d1_1, d2_1, color="gold", s=5, label="Warning 1", alpha=0.7)
        ax.scatter(d1_2, d2_2, color="darkorange", s=5, label="Warning 2", alpha=0.7)
        ax.scatter(d1_3, d2_3, color="red", s=5, label="Warning 3", alpha=0.7)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    
    fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5)

    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, 0, 9), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

# box plot *****************************

def box_warning_pupil_d(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    subtaskData = all[userId, uiId]
    
    data = []
    for sub in subtaskData:
        subData = []
        for i in range(4):
            mask = sub["warning"] == i
            if np.any(mask):
                arr = sub["lg_pupil_d"][mask]
                arr = arr[arr != 0]  # Exclude 0 values
                arr = arr[arr != -1]
                subData.append(arr)  # Exclude -1 and 0 values
            else:
                subData.append(np.array([]))
        data.append(subData)

    box_plot_3x3(data, "Pupil Size Diff & Warning: User-{0:02d}, UI-{1} ({2})".format(userId, UI[uiId], subtaskData[0]["taskOrder"]), "Pupil Size Diff [mm]", "Warning", subtaskData, xticklabel=range(4), ylim=(2, 8), figsize=figsize, save=save, pdf=pdf)

def box_sum_warning_pupil_d(userId, uiId, figsize=FIGSIZE, save=SAVE, pdf=None):
    userData = all[userId]
    
    data = []
    for subtaskData in userData:
        subData = [[] for _ in range(4)]  # Create a list for each warning level
        for sub in subtaskData:
            for i in range(4):
                mask = sub["warning"] == i
                if np.any(mask):
                    arr = sub["lg_pupil_d"][mask]
                    arr = arr[arr != 0]  # Exclude 0 values
                    arr = arr[arr != -1]
                    subData[i].extend(arr)  # Sum of pupil size differences
        data.append(subData)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.9, wspace=0.11, bottom=0.08)
    fig.suptitle("Sum of Pupil Size Diff & Warning: User-{0:02d}".format(userId))
    axs = axs.flatten()
    xticklabel = range(4)

    for i, (ax, d) in enumerate(zip(axs, data)):
        ax.boxplot(d, patch_artist=True, notch=True, showmeans=True, meanline=True, tick_labels=xticklabel)
        test = samples_test_ind_list(d)
        txt = ""
        for v in range(4):
            for u in range(v+1, 4):
                if test[v][u][0]:
                    txt += "{0}-{1}, ".format(v, u)
        if txt:
            ax.set_xlabel("*: " + txt[:-2])
        ax.set_title("UI-{0} ({1})".format(UI[i], userData[i][0]["taskOrder"]))
        ax.set_ylim(0, 9)
    fig.supxlabel("Warning", x=0.5, y=0.01)
    fig.supylabel("Pupil Size Diff [mm]", x=0.01, y=0.5)

    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, userId, 9), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
        pass
    
    plt.close()

def box_cognition_time_robot_num(figsize=FIGSIZE, save=SAVE, pdf=None):

    if pdf:
        pdf_path = PPATH.format("box_cognition_time_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if not sub["collision_flag"]: continue

                pw = 0
                pt = sub["time"][0]
                flag = False
                center = np.array(TASK_TILE_CENTER[sub["state"]])
                for t, w, p, c in zip(sub["time"], sub["warning"], sub["pos"], sub["collision"]):
                    if pw == 0 and w > 0:
                        pt = t
                        flag = True
                        # udata[ROBOT_NUM[sub["state"]]-1].append((pw, w, pt-sub["time"][0]))
                    
                    if flag and (distance(p, center) >= 0.707):
                        if t != pt:
                            data[0][ROBOT_NUM[sub["state"]]][uiId].append(t-pt)
                            data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(t-pt)
                        flag = False
                    
                    pw = w
    
    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Cognition Time by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Cognition Time [s]", xticklabel=UI, ylim=(-0.5, 5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)
        else:
            continue
            title = "Box Plot of Cognition Time by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Cognition Time [s]", xticklabel=UI, ylim=(-0.5, 5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_collision_robot_num(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_collision_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)]  # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskCollision"])
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskCollision"])
                else:
                    data[0][0][uiId].append(sub["taskCollision"])
                    data[sub["taskOrder"]][0][uiId].append(sub["taskCollision"])
    
    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Collision Count by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Collision Count", xticklabel=UI, ylim=(-1, 16), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=True)
        else:
            continue
            title = "Box Plot of Collision Count by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Collision Count", xticklabel=UI, ylim=(-1, 16), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)
    
    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_time_robot_num(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_time_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskTime"])
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskTime"])
                else:
                    data[0][0][uiId].append(sub["taskTime"])
                    data[sub["taskOrder"]][0][uiId].append(sub["taskTime"])
    
    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Task Time by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Task Time [s]", xticklabel=UI, ylim=(-10, 130), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=True)
        else:
            continue
            title = "Box Plot of Task Time by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Task Time [s]", xticklabel=UI, ylim=(-10, 130), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_time_robot_num_filter(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_ontile_percent_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                tile_pos = TASK_TILE_CENTER[sub["state"]]
                diff = np.abs(sub["pos"] - tile_pos)
                within_x = diff[:, 0] <= 0.5
                within_z = diff[:, 2] <= 0.5
                percent = np.sum(within_x & within_z) / len(sub["pos"])
                time = percent

                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(time)
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(time)
                else:
                    data[0][0][uiId].append(time)
                    data[sub["taskOrder"]][0][uiId].append(time)

    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of On-tile Percent by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Task Time [s]", xticklabel=UI, ylim=(-0.1, 1.1), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=True)
        else:
            title = "Box Plot of On-tile Percent by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Task Time [s]", xticklabel=UI, ylim=(-0.1, 1.1), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_robot_distance_robot_num(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_robot_distance_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                dist = min(sub["min_dist"])  # Limit distance to 6m
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(dist)
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(dist)
                else:
                    data[0][0][uiId].append(dist)
                    data[sub["taskOrder"]][0][uiId].append(dist)

    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Robot Distance by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Robot Distance [m]", xticklabel=UI, ylim=(-0.5, 4), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=True)
        else:
            continue
            title = "Box Plot of Robot Distance by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Robot Distance [m]", xticklabel=UI, ylim=(-0.5, 4), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_time_robot_num_until_collision(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_time_robot_num_until_collision")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                idx = np.where(sub["collision"] == True)[0]
                if len(idx) == 0:
                    tasktime = sub["taskTime"]
                else:
                    tasktime = sub["time"][idx[0]] - sub["time"][0]
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(tasktime)
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(tasktime)
                else:
                    data[0][0][uiId].append(tasktime)
                    data[sub["taskOrder"]][0][uiId].append(tasktime)

    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Task Time by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Task Time [s]", xticklabel=UI, ylim=(-10, 130), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=True)
        else:
            title = "Box Plot of Task Time by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Task Time [s]", xticklabel=UI, ylim=(-10, 130), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_mistake_robot_num(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_mistake_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskMistake"])
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskMistake"])
                else:
                    data[0][0][uiId].append(sub["taskMistake"])
                    data[sub["taskOrder"]][0][uiId].append(sub["taskMistake"])
    
    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Mistake Count by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Mistake Count", xticklabel=UI, ylim=(-0.5, 3.5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=True)
        else:
            title = "Box Plot of Mistake Count by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Mistake Count", xticklabel=UI, ylim=(-0.5, 3.5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_cognition_time_ui(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_cognition_time_ui")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if not sub["collision_flag"]: continue

                pw = 0
                pt = sub["time"][0]
                flag = False
                center = np.array(TASK_TILE_CENTER[sub["state"]])
                for t, w, p, c in zip(sub["time"], sub["warning"], sub["pos"], sub["collision"]):
                    if pw == 0 and w > 0:
                        pt = t
                        flag = True
                    
                    if flag and (distance(p, center) >= 0.707):
                        if t != pt:
                            data[0][uiId][ROBOT_NUM[sub["state"]]].append(t-pt)
                            data[sub["taskOrder"]][uiId][ROBOT_NUM[sub["state"]]].append(t-pt)
                        flag = False
                    
                    pw = w

    titles = UI
    for i in range(5):
        if i == 0:
            title = "Box Plot of Cognition Time by UI (All)"
            box_plot_2x2(data[i], title, "Robot Num", "Cognition Time [s]", xticklabel=range(4), ylim=(-0.5, 5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=range(4))
        else:
            title = "Box Plot of Cognition Time by UI (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "Robot Num", "Cognition Time [s]", xticklabel=range(4), ylim=(-0.5, 5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=range(4))

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_collision_ui(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_collision_ui")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][uiId][ROBOT_NUM[sub["state"]]].append(sub["taskCollision"])
                    data[sub["taskOrder"]][uiId][ROBOT_NUM[sub["state"]]].append(sub["taskCollision"])
                else:
                    data[0][uiId][0].append(sub["taskCollision"])
                    data[sub["taskOrder"]][uiId][0].append(sub["taskCollision"])
    
    titles = UI
    for i in range(5):
        if i == 0:
            title = "Box Plot of Collision Count by UI (All)"
            box_plot_2x2(data[i], title, "Robot Num", "Collision Count", xticklabel=range(4), ylim=(-1, 16), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=list(range(4)))
        else:
            title = "Box Plot of Collision Count by UI (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "Robot Num", "Collision Count", xticklabel=range(4), ylim=(-1, 16), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=list(range(4)))

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_time_ui(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_time_ui")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][uiId][ROBOT_NUM[sub["state"]]].append(sub["taskTime"])
                    data[sub["taskOrder"]][uiId][ROBOT_NUM[sub["state"]]].append(sub["taskTime"])
                else:
                    data[0][uiId][0].append(sub["taskTime"])
                    data[sub["taskOrder"]][uiId][0].append(sub["taskTime"])
    
    titles = UI
    for i in range(5):
        if i == 0:
            title = "Box Plot of Task Time by UI (All)"
            box_plot_2x2(data[i], title, "Robot Num", "Task Time [s]", xticklabel=range(4), ylim=(-10, 130), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=list(range(4)))
        else:
            title = "Box Plot of Task Time by UI (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "Robot Num", "Task Time [s]", xticklabel=range(4), ylim=(-10, 130), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=list(range(4)))

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_mistake_ui(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_mistake_ui")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][uiId][ROBOT_NUM[sub["state"]]].append(sub["taskMistake"])
                    data[sub["taskOrder"]][uiId][ROBOT_NUM[sub["state"]]].append(sub["taskMistake"])
                else:
                    data[0][uiId][0].append(sub["taskMistake"])
                    data[sub["taskOrder"]][uiId][0].append(sub["taskMistake"])

    
    titles = UI
    for i in range(5):
        if i == 0:
            title = "Box Plot of Mistake Count by UI (All)"
            box_plot_2x2(data[i], title, "Robot Num", "Mistake Count", xticklabel=range(4), ylim=(-0.5, 3.5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=list(range(4)))
        else:
            title = "Box Plot of Mistake Count by UI (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "Robot Num", "Mistake Count", xticklabel=range(4), ylim=(-0.5, 3.5), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=list(range(4)))

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_pupil_d_robot_num(figsize=FIGSIZE, save=SAVE, pdf=None):
    if pdf:
        pdf_path = PPATH.format("box_pupil_d_robot_num")
        pdf = PdfPages(pdf_path)
    
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)] # 4 robot numbers, 4 ui states

    for userData in all:
        if userData[0][0]["userId"] in [2, 4, 14]:
            continue
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    arr = sub["lg_pupil_d"]
                    arr = arr[arr != 0]  # Exclude 0 values
                    arr = arr[arr != -1]  # Exclude -1 values
                    if len(arr) != 0:
                        data[0][ROBOT_NUM[sub["state"]]][uiId].extend(arr)
                        data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].extend(arr)
                else:
                    arr = sub["lg_pupil_d"]
                    arr = arr[arr != 0]  # Exclude 0 values
                    arr = arr[arr != -1]  # Exclude -1 values
                    if len(arr) != 0:
                        data[0][0][uiId].extend(arr)
                        data[sub["taskOrder"]][0][uiId].extend(arr)

    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    for i in range(5):
        if i == 0:
            title = "Box Plot of Pupil Size Diff by Robot Number (All)"
            box_plot_2x2(data[i], title, "UI", "Pupil Size Diff [mm]", xticklabel=UI, ylim=(0, 9), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=UI)
        else:
            title = "Box Plot of Pupil Size Diff by Robot Number (Task Order: {0})".format(i)
            box_plot_2x2(data[i], title, "UI", "Pupil Size Diff [mm]", xticklabel=UI, ylim=(0, 9), figsize=FIGSIZE, titles=titles, save=save, pdf=pdf, rel=False, tlabel=UI)

    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

def box_plot_2x2(data, title, xlabel, ylabel, xticklabel, ylim, titles, figsize=FIGSIZE, save=SAVE, pdf=None, rel=True, tlabel=UI):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.9, wspace=0.11, bottom=0.08)

    axs = axs.flatten()

    for i, (ax, d) in enumerate(zip(axs, data)):
        ax.boxplot(d, patch_artist=True, notch=False, showmeans=True, meanline=True, tick_labels=xticklabel)
        ax.set_title(titles[i])
        ax.set_ylim(*ylim)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        if rel: 
            res = samples_test_rel_list(d, n_parametric=True)
            group = friedmanchisquare(*d)
        else:
            res = samples_test_ind_list(d, n_parametric=True)
            if len(d[0]) == 0:
                group = {"statistic": 0, "pvalue": 1.0}
            else:
                group = kruskal(*d)

        txt = ""
        if len(d[0]) != 0:
            print("id: {0} | s:{1}, p: {2} | ".format(i, group.statistic, group.pvalue), end="")
            ave = [str(np.mean(x)) for x in d]
            print(", ".join(ave), end="")
        for u in range(4):
            for v in range(u+1, 4):
                if res[u][v][0]:
                    # txt += "{0}-{1}({2:.2f}, {3:.2f}), ".format(tlabel[u], tlabel[v], res[u][v][2], res[u][v][1])
                    txt += "{0}-{1}({2:.2f}), ".format(tlabel[u], tlabel[v], res[u][v][2])
                # print("{0}-{1}({2}), ".format(tlabel[u], tlabel[v], res[u][v][2]), end="")
        print()

        if txt:
            pass
            # ax.set_xlabel("*: "+txt[:-2])

    fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def box_plot_3x3(data, title, xlabel, ylabel, subtaskData, xticklabel, ylim, figsize=FIGSIZE, save=SAVE, pdf=None):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.9, wspace=0.11, bottom=0.08)

    axs = axs.flatten()

    for i, (ax, d) in enumerate(zip(axs, data)):
        ax.boxplot(d, patch_artist=True, notch=True, showmeans=True, meanline=True, tick_labels=xticklabel)
        ax.set_title("Subtask {0}".format(subtaskData[i]["label"]))
        ax.set_ylim(*ylim)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
    
    fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5)

    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, 0, 9), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

def box_plot(data, title, xlabel, ylabel, xticklabel, ylim, figsize=FIGSIZE, save=SAVE, pdf=None):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.9, wspace=0.11, bottom=0.08)

    ax.boxplot(data, patch_artist=True, notch=True, showmeans=True, meanline=True, tick_labels=xticklabel)
    ax.set_ylim(*ylim)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)

    fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5)

    if save:
        os.makedirs(f"pic/{save}", exist_ok=True)
        plt.savefig(PATH.format(save, 0, 9), dpi=DPI)
    elif pdf:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close()

# group plot *****************************

def group_time_collision(figsize=FIGSIZE, save=SAVE, pdf=False):
    if pdf:
        pdf_path = PPATH.format("group_time_collision_gender-game")
        pdf = PdfPages(pdf_path)
    
    collision = [[[] for _ in range(4) ] for _ in range(4)]
    time = [[[] for _ in range(4) ] for _ in range(4)]

    for userData in all:
        for uiId in range(4):
            data = userData[uiId]
            c = [list() for _ in range(4)]
            t = [list() for _ in range(4)]
            for sub in data:
                if not sub["collision_flag"]:
                    c[0].append(sub["taskCollision"])
                    t[0].append(sub["taskTime"])
                elif ROBOT_NUM[sub["state"]] == 1:
                    c[1].append(sub["taskCollision"])
                    t[1].append(sub["taskTime"])
                elif ROBOT_NUM[sub["state"]] == 2:
                    c[2].append(sub["taskCollision"])
                    t[2].append(sub["taskTime"])
                else:
                    c[3].append(sub["taskCollision"])
                    t[3].append(sub["taskTime"])

            for j in range(4):
                collision[j][uiId].append(c[j])
                time[j][uiId].append(t[j])

    markers1 = [(m, c) for m, c in itertools.product(["o", "s", "^", "D", "x"], ["#87CEEB", "#4169E1", "#191970", "#00732C", "#07E104"])]
    markers2 = [(m, c) for m, c in itertools.product(["o", "s", "^", "D", "x"], ["#FFB6C1", "#FF69B4", "#C71585", "#FB6A16", "#FFA114"])]

    for i in range(4):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        fig.subplots_adjust(hspace=0.3, left=0.06, right=0.98, top=0.9, wspace=0.11, bottom=0.09)
        fig.suptitle("Group Time & Collision Count: Robot Num {0}".format(i))

        axs = axs.flatten()
        for j, ax in enumerate(axs):
            t = [ d for user in time[i][j] for d in user ]
            c = [ d for user in collision[i][j] for d in user ]
            corr = get_corr([t], [c], axis=1)[0]
            ax.set_title("UI-{0}: {1:.2f}".format(UI[j], corr))

            genre = {"mobile": "o", "console": "s", "vr": "D", "none": "x"}
            gender = {"m": "blue", "f": "red", "other": "green"}
            duration = ["x", "o", "s", "D"]
            for userId, (t, c) in enumerate(zip(time[i][j], collision[i][j])):
                ax.scatter(t, c, marker=genre[attr["genre"][userId]], color=gender[attr["gender"][userId]], s=20, alpha=0.7, label="__nolegend__")

            ax.set_xlim(-10, 130)
            ax.set_ylim(-1, 16)
            # ax.grid(True)

        # for k, m in zip(["0", "-59", "-119", "120-"], duration):
        #     axs[0].scatter([], [], marker=m, color="blue", s=20, label=k)

        for k, m in genre.items():
            axs[0].scatter([], [], marker=m, color="blue", s=20, label=k)

        fig.legend(loc='lower left', ncol=4, bbox_to_anchor=(0, 0), fontsize='small', title="most frequent game genre")
        fig.supxlabel("Task Time [s]", x=0.5, y=0.01)
        fig.supylabel("Collision Count", x=0.01, y=0.5)

        if save:
            os.makedirs(f"pic/{save}", exist_ok=True)
            plt.savefig(PATH.format(save, 0, 9), dpi=DPI)
        elif pdf:
            pdf.savefig(fig)
        else:
            plt.show()
    
        plt.close()
    
    if pdf:
        pdf.close()
        print("Saved PDF: {0}".format(pdf_path))

# tool functions *****************************

def get_max_min(attr):
    max_val = max([max([max([data[attr] if not isinstance(data[attr], np.ndarray) else max(data[attr]) for data in uidata ]) for uidata in userData]) for userData in all])
    min_val = min([min([min([data[attr] if not isinstance(data[attr], np.ndarray) else min(data[attr]) for data in uidata ]) for uidata in userData]) for userData in all])
    print("{0} | Max: {1}, Min: {2}".format(attr, max_val, min_val))


def collision_plot(figsize=FIGSIZE, save=SAVE):
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)]  # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskCollision"])
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskCollision"])
                else:
                    data[0][0][uiId].append(sub["taskCollision"])
                    data[sub["taskOrder"]][0][uiId].append(sub["taskCollision"])
    
    titles = ["No Collision", "Robot 1", "Robot 2", "Robot 3"]
    if save:
        save = "collision"

    # box_3x1(data[0][1:4], title="The number of collision", ylabel="the number of collision", ylim=(-1, 20), yticks=range(0, 16, 3), save=save)
    box_plot(data[0][1:4], ylabel="The number of collision", ylim=(-1, 20), yticks=range(0, 16, 3), xticklabel=titles[1:], legends=UI, save=save)

def time_plot(figsize=FIGSIZE, save=SAVE):
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)]  # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskTime"])
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(sub["taskTime"])
                else:
                    data[0][0][uiId].append(sub["taskTime"])
                    data[sub["taskOrder"]][0][uiId].append(sub["taskTime"])

    titles = ["Robot 0", "Robot 1", "Robot 2", "Robot 3"]
    if save:
        save = "time"

    # box_1x4(data[0], title="Task completion time", ylabel="task completion time [s]", ylim=(-10, 150), yticks=range(0, 130, 20), save=save)
    box_plot(data[0], ylabel="Task completion time [s]", ylim=(-10, 150), yticks=range(0, 130, 20), xticklabel=titles, legends=UI, save=save)
    
def box_2x2(data, ylabel, xticklabel, ylim, yticks, titles, figsize=FIGSIZE, save=SAVE):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.subplots_adjust(hspace=0.2, left=0.06, right=0.98, top=0.97, wspace=0.11, bottom=0.03)

    axs = axs.flatten()

    for i, (ax, d) in enumerate(zip(axs, data)):
        ax.boxplot(d, tick_labels=xticklabel, showmeans=True, medianprops={'color':'orange', 'linewidth':3, 'linestyle':'-'})
        ax.set_title(titles[i])
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)

    # fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
    else:
        plt.show()
    plt.close()

def box_1x4(data, title, ylabel, ylim, yticks, xticklabel=None, titles=None, figsize=(12, 6), save=SAVE):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize)
    fig.subplots_adjust(hspace=0.2, left=0.07, right=0.995, top=0.95, wspace=0.04, bottom=0.04)
    # fig.subplots_adjust(hspace=0.2, left=0.06, right=0.995, top=0.895, wspace=0.04, bottom=0.04)

    axs = axs.flatten()

    # fig.suptitle()

    if not titles:
        titles = ["Robot 0", "Robot 1", "Robot 2", "Robot 3"]
    
    if not xticklabel:
        xticklabel = UI
        # xticklabel = [ l if i%2 == 0 else "\n"+l for i, l in enumerate(xticklabel) ]  # Add newline for better visibility

    for i, (ax, d) in enumerate(zip(axs, data)):
        ax.boxplot(d, tick_labels=xticklabel, showmeans=True, medianprops={'color':'orange', 'linewidth':3, 'linestyle':'-'})
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(titles[i], fontsize=16)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        if i != 0:
            ax.set_yticklabels([])

    # fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5, fontsize=16)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
    else:
        plt.show()
    plt.close()

def robot_distance_plot(figsize=FIGSIZE, save=SAVE):
    data = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(5)]  # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(4):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                dist = min(sub["min_dist"])  # Limit distance to 6m
                if sub["collision_flag"]:
                    data[0][ROBOT_NUM[sub["state"]]][uiId].append(dist)
                    data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(dist)
                else:
                    data[0][0][uiId].append(dist)
                    data[sub["taskOrder"]][0][uiId].append(dist)   
    
    if save:
        save = "robot_distance"
    # box_3x1(data[0][1:4], title="The closest distance to robots", ylabel="the closest distance to robots [m]", ylim=(-0.4, 4), yticks=np.arange(0, 3.1, 0.5), save=save)
    box_plot(data[0][1:4], ylabel="The closest distance to robots [m]", ylim=(-0.4, 4), yticks=np.arange(0, 3.1, 0.5), xticklabel=["Robot 1", "Robot 2", "Robot 3"], legends=UI, save=save)

def cognition_time_plot(figsize=FIGSIZE, save=SAVE):
    data = [[[[] for _ in range(3)] for _ in range(4)] for _ in range(5)]  # 4 robot numbers, 4 ui states

    for userData in all:
        for uiId in range(3):
            subtaskData = userData[uiId]
            for sub in subtaskData:
                if not sub["collision_flag"]: continue

                pw = 0
                pt = sub["time"][0]
                flag = False
                center = np.array(TASK_TILE_CENTER[sub["state"]])
                for t, w, p, c in zip(sub["time"], sub["warning"], sub["pos"], sub["collision"]):
                    if pw == 0 and w > 0:
                        pt = t
                        flag = True
                        # udata[ROBOT_NUM[sub["state"]]-1].append((pw, w, pt-sub["time"][0]))
                    
                    if flag and (distance(p, center) >= 0.707):
                        if t != pt:
                            data[0][ROBOT_NUM[sub["state"]]][uiId].append(t-pt)
                            data[sub["taskOrder"]][ROBOT_NUM[sub["state"]]][uiId].append(t-pt)
                        flag = False
                    
                    pw = w

    if save:
        save = "cognition_time"

    # box_3x1(data[0][1:4], title="Time to leave task area after warning", ylabel="time to leave task area after warning[s]", xticklabel=UI[0:3], ylim=(-0.5, 5), yticks=np.arange(0, 4.1, 1), save=save)
    box_plot(data[0][1:4], ylabel="Time to leave task area after warning [s]", ylim=(-0.5, 5), yticks=np.arange(0, 4.1, 1), xticklabel=["Robot 1", "Robot 2", "Robot 3"], legends=UI[0:3], save=save)

def box_3x1(data, title, ylabel, ylim, yticks, xticklabel=False, titles=False, figsize=(12, 6), save=SAVE):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    fig.subplots_adjust(hspace=0.2, left=0.07, right=0.99, top=0.95, wspace=0.04, bottom=0.05)

    axs = axs.flatten()
    if not titles:
        titles = ["Robot 1", "Robot 2", "Robot 3"]
    if not xticklabel:
        xticklabel = UI

    # fig.suptitle(title)
    for i, (ax, d) in enumerate(zip(axs, data)):
        ax.boxplot(d, tick_labels=xticklabel, showmeans=True, medianprops={'color':'orange', 'linewidth':3, 'linestyle':'-'})
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(titles[i], fontsize=16)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        if i != 0:
            ax.set_yticklabels([])
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)

    # fig.supxlabel(xlabel, x=0.5, y=0.01)
    fig.supylabel(ylabel, x=0.01, y=0.5, fontsize=16)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
    else:
        plt.show()
    plt.close()

def questionnaire_plot(save=SAVE):
    attr = ["easy", "annoy", "safe"]
    plot_data = [list(data[a][:, 0:3].T) if a != "safe" else list(data[a].T) for a in attr]

    labels = ["Easy to understand", "Intrusive", "Safety perception"]

    if save:
        save = "questionnaire"

    box_plot(plot_data, ylabel="", ylim=(0.5, 9), yticks=range(1, 8), xticklabel=labels, legends=UI, save=save)

def nasatlx_plot(save=SAVE):
    attr = [["mental", "physical", "temporal", "performance"], ["effort", "frustration", "score"]]
    plot_data = [[list(data[a].T) for a in r]for r in attr]

    labels = [["Mental demand", "Physical demand", "Temporal demand", "Performance"], ["Effort", "Frustration", "Overall score"]]

    if save:
        save = "nasatlx"

    box_plot_2x1(plot_data, ylabel="", ylim=(-10, 150), yticks=range(0, 101, 20), xticklabel=labels, legends=UI, save=save)

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
    else:
        plt.show()
    plt.close()

def box_plot_2x1(data, ylabel, ylim, yticks, xticklabel=False, legends=False, figsize=(12, 10), save=SAVE):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    fig.subplots_adjust(hspace=0.13, left=0.048, right=0.99, top=0.94, wspace=0.04, bottom=0.06)

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
        # ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=16)
        
    
    for k, g in enumerate(legends):
        boxes.append(plt.Rectangle((0, 0), 1, 1, color=colors[k], ec="black", alpha=0.5))
        labels.append(g)
    
    
    axs[0].legend(boxes, labels, loc='lower center', ncol=len(legends), fontsize=18, bbox_to_anchor=(0.5, 1))

    fig.supylabel(ylabel, x=0.01, y=0.5, fontsize=18)

    if save:
        os.makedirs(f"pic", exist_ok=True)
        plt.savefig(FIG.format(save), dpi=DPI)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        all = load_all()
        attr = load_subject_attr()
        data = load_subject()
        # total, average = calculate_change(i, j, "pos")
        # print(i, j, "Total change: {0}, Average change: {1}".format(total, average))
        # calculate_change(i, j, "qua")
        # pie_plot(i, j, "obj", OBJ_LABEL)
        # barh_plot(i, j, "obj", OBJ_LABEL)
        # robot_distance(3, 0)
        # barh_plot(3, 0, "obj", OBJ_LABEL)
        # barh_plot_all(3, "obj", OBJ_LABEL)
        # rot_y_plot(3, 0)
        # robot_distance(3, 0)
        # rot_y_diff_plot(3, 0)
        # save_pdf(bpm)
        # save_pdf(pos_xz, args=dict())
        # rot_y_diff(3, 3)
        # rot_y_diff_n(3, 3, 10)
        # map_func(pos_xz, args=dict())
        # pos_xz(3, 0)
        # twin_xz_roty_diff_n(3, 0, 5, legends=["Position XZ", "Rotation Y"])
        # map_func(pos_xz_diff_center)
        # map_func(gaze_diff_n, args=dict(n=5))
        # map_func(robot_distance_diff_n, args=dict(n=5))
        # map_func(eye_open)
        # map_func(pupil_size)
        # map_func(pupil_pos_diff_n, args=dict(n=5))

        # box_warning_pupil_d(3, 0)
        # box_sum_warning_pupil_d(3, 0)
        # map_func(box_sum_warning_pupil_d, uiIdRange=range(0, 1))
        # get_max_min("taskMistake")
        # get_max_min("taskCollision")
        # get_max_min("taskTime")
        # get_max_min("min_dist")

        # save_pdf(barh_plot_all, args=dict(attr="obj"), uiIdRange=range(0, 1))
        # save_pdf(pos_xz_diff_n, args=dict(n=60))
        # save_pdf(pos_xz)
        # save_pdf(rot_y_diff_n, args=dict(n=60))
        # save_pdf(bpm)
        # save_pdf(robot_distance)
        # save_pdf(robot_distance_diff_n, args=dict(n=60))
        # save_pdf(rot_y_diff)
        # save_pdf(rot_y)
        # save_pdf(twin_xz_roty_diff_n, args=dict(n=60, legends=["Position XZ", "Rotation Y"]))
        # save_pdf(subtask_time, uiIdRange=range(0, 1))
        # save_pdf(collision_count, uiIdRange=range(0, 1))
        # save_pdf(mistake_count, uiIdRange=range(0, 1))
        # save_pdf(twin_robot_distance_xz_diff_center, args=dict(legends=["Robot Distance", "Position XZ Diff Center"]))
        # save_pdf(pos_xz_diff_center)
        # save_pdf(gaze_diff_n, args=dict(n=1))
        # save_pdf(eye_open)
        # save_pdf(pupil_size)
        # save_pdf(scatter_pos_xz_diff_center_robot_distance)
        # save_pdf(scatter_pupil_size_pos_xz_diff_center, args=dict(n=5))
        # save_pdf(scatter_pupil_d_robot_distance, args=dict(n=60))
        # save_pdf(scatter_warning_pupil_d, args=dict(n=0))
        # save_pdf(box_warning_pupil_d, uiIdRange=range(3, 4), name="_NO")
        # save_pdf(box_sum_warning_pupil_d, uiIdRange=range(1), name="")
        # group_time_collision(pdf=True)
        # box_mistake_robot_num(pdf=True)
        # box_collision_ui(pdf=True)
        # box_time_ui(pdf=True)
        # box_mistake_ui(pdf=True)
        # box_pupil_d_robot_num(pdf=True)

        # box_collision_robot_num(pdf=True)
        # box_time_robot_num(pdf=True)
        # box_robot_distance_robot_num(pdf=True)
        # box_cognition_time_robot_num(pdf=True)


        collision_plot(save=True)
        time_plot(save=True)
        robot_distance_plot(save=True)
        cognition_time_plot(save=True)
        questionnaire_plot(save=True)
        nasatlx_plot(save=True)

        # questionnaire_plot(save=False)
        # robot_distance_plot(save=False)
        # time_plot(save=False)
        # cognition_time_plot(save=False)
        # nasatlx_plot(save=False)

        pass
    except Exception as e:
        traceback.print_exc()